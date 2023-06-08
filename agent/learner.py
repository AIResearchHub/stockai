

import torch
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

import numpy as np
import threading
import time
import random
from copy import deepcopy

from .actor import Actor
from .replay_buffer import ReplayBuffer

from model import Critic, Policy
from utils import read_context, get_context


class Learner:
    """
    Main class used to train the agent. Called by rpc remote.
    Call run() to start the main training loop.

    Parameters:

    buffer_size (int): The size of the buffer in ReplayBuffer
    batch_size (int): Batch size for training
    n_accumulate (int): Number of times to accumulate gradients before optimizer step
    tickers (List): A list of n tickers e.g. ["AAPL", "GOOGL", "BAC"]
    mock_data (bool): If true then only one news paragrah per day for each data for test run
    vocab_size (int): Vocabulary size for transformer
    n_layers (int): Number of layers in transformer
    d_model (int): Dimensions of the model
    n_head (int): Number of attention heads in transformer
    n_cos (int): Number of cosine samples for each tau in IQN
    n_tau (int): Number of tau samples for IQN each representing a value for a percentile
    state_len (int): Length of recurrent state
    n_step (int): N step returns see https://paperswithcode.com/method/n-step-returns
    burnin_len (int): Length of burnin, concept from R2D2 paper
    rollout_len (int): Length of rollout, concept from R2D2 paper

    """
    epsilon = 1
    epsilon_min = 0.2
    epsilon_decay = 0.0001

    lr = 1e-4
    gamma = 0.99

    tau = 1e-2
    save_every = 100

    # TD3 noise
    policy_noise = 0.2
    noise_clip = 0.5

    def __init__(self,
                 cls,
                 buffer_size,
                 batch_size,
                 n_accumulate,
                 tickers,
                 mock_data,
                 vocab_size,
                 max_len,
                 n_layers,
                 d_model,
                 n_head,
                 n_cos,
                 n_tau,
                 n_p,
                 state_len,
                 n_step,
                 burnin_len,
                 rollout_len
                 ):
        # torch.manual_seed(0)
        # np.random.seed(0)
        # random.seed(0)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_accumulate = n_accumulate
        self.tickers = tickers

        self.vocab_size = vocab_size
        self.max_len = max_len

        self.d_model = d_model
        self.state_len = state_len
        self.n_tau = n_tau
        self.n_p = n_p

        # Twin Delayed DDPG (TD3)

        model = nn.DataParallel(
            Critic(cls=cls,
                   vocab_size=vocab_size,
                   max_len=max_len,
                   n_layers=n_layers,
                   d_model=d_model,
                   n_head=n_head,
                   n_cos=n_cos
                   )
        ).cuda()

        policy = nn.DataParallel(
            Policy(cls=cls,
                   vocab_size=vocab_size,
                   max_len=max_len,
                   n_layers=n_layers,
                   d_model=d_model,
                   n_head=n_head
                   )
        ).cuda()

        self.model1 = deepcopy(model)
        self.model2 = deepcopy(model)
        self.target_model1 = deepcopy(model)
        self.target_model2 = deepcopy(model)
        self.policy = deepcopy(policy)
        self.target_policy = deepcopy(policy)
        self.eval_policy = deepcopy(policy)

        # set model modes
        self.model1.train()
        self.model2.train()
        self.target_model1.eval()
        self.target_model2.eval()
        self.policy = self.policy.eval()
        self.target_policy = self.target_policy.eval()
        self.eval_policy = self.eval_policy.eval()

        # contexts
        self.contexts = read_context(tickers=tickers,
                                     mock_data=mock_data,
                                     max_len=max_len,
                                     )

        # locks
        self.lock = mp.Lock()
        self.lock_model = mp.Lock()

        # hyper-parameters
        self.burnin_len = burnin_len
        self.rollout_len = rollout_len
        self.block_len = burnin_len + rollout_len
        self.n_step = n_step
        self.gamma = self.gamma ** n_step

        # optimizer and loss functions
        self.optimizer1 = optim.Adam(self.model1.parameters(), lr=self.lr)
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.nll_loss = nn.NLLLoss(ignore_index=0)

        # queues
        self.sample_queue = mp.Queue()
        self.batch_queue = mp.Queue()
        self.priority_queue = mp.Queue()

        self.batch_queue = mp.Queue(8)
        self.priority_queue = mp.Queue(8)

        # params, batched_data (feeds batch), pending_rpcs (answer calls)
        self.batch_data = []

        # start replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          batch_size=batch_size*n_accumulate,
                                          block_len=burnin_len+rollout_len,
                                          max_len=max_len,
                                          d_model=d_model,
                                          state_len=max_len,
                                          n_step=n_step,
                                          gamma=self.gamma,
                                          contexts=self.contexts,
                                          sample_queue=self.sample_queue,
                                          batch_queue=self.batch_queue,
                                          priority_queue=self.priority_queue
                                          )

        # start actors
        self.future1 = Future()
        self.future2 = Future()

        self.pending_rpc = None
        self.await_rpc = False

        self.actor_rref = self.spawn_actor(learner_rref=RRef(self),
                                           tickers=self.tickers,
                                           d_model=self.d_model,
                                           state_len=self.state_len
                                           )

        self.update_steps = 0

    @staticmethod
    def spawn_actor(learner_rref, tickers, d_model, state_len):
        """
        Start actor by calling actor.remote().run()
        Actors communicate with learner through rpc and RRef

        Parameters:
        learner_rref (RRef): learner RRef for actor to reference the learner
        tickers (List[2]): A list of tickers e.g. ["AAPL", "GOOGL"]
        d_model (int): Dimension of model
        state_len (int): Length of recurrent state

        Returns:
        actor_rref (RRef): to reference the actor from the learner
        """
        actor_rref = rpc.remote("actor",
                                Actor,
                                args=(learner_rref,
                                      tickers,
                                      d_model,
                                      state_len
                                      ),
                                timeout=0
                                )
        actor_rref.remote().run()

        return actor_rref

    @async_execution
    def queue_request(self, *args):
        """
        Called by actor asynchronously to queue requests

        Returns:
        future (Future.wait): Halts until value is ready
        """
        future = self.future1.then(lambda f: f.wait())
        with self.lock:
            self.pending_rpc = args

        return future

    @async_execution
    def return_episode(self, episode):
        """
        Called by actor to asynchronously to return completed Episode
        to Learner

        Returns:
        future (Future.wait): Halts until value is ready
        """
        future = self.future2.then(lambda f: f.wait())
        self.sample_queue.put(episode)
        self.await_rpc = True

        return future

    def get_policy(self, x, state):
        """
        Function to get actor from eval_model which is constantly updated
        during training

        Parameters:
        x (Tensor[1, 1], Tensor[1, max_len]): allocation value and tokens to pass into model
        state (Tensor[1, state_len, d_model]): recurrent state to pass into model

        Returns:
        action (float): action value with interval (-1, 1)
        state (Array[1, state_len, d_model): next recurrent state

        """
        assert x[0].shape == (1, 1)
        assert x[1].shape == (1, self.max_len)

        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        with self.lock_model:
            with torch.no_grad():
                action, state = self.eval_policy.forward(x=x, state=state)

        if random.random() <= self.epsilon:
            return random.uniform(-1, 1), state.detach().cpu().numpy()

        action = action.cpu().squeeze().numpy().item()
        state = state.detach().cpu().numpy()

        return action, state

    def get_action(self, alloc, timestamp, tickers, state):
        """
        Get action function that turns all the env inputs into tensor
        and calls get_policy to get action value
        If state is None, then return 0. as action and initialize state
        through self.module.init_state()

        Parameters:
        alloc (float): allocation value
        timestamp (datetime.datetime): timestamp of current time step
        tickers (List[2]): List containing tickers e.g. ["AAPL", "GOOGL"]
        state (Array[1, state_len, d_model]): recurrent state

        Returns:
        action (float): Action value
        state (Array[1, state_len, d_model]): Next recurrent state

        """
        if state is None:
            return 0., self.eval_policy.module.init_state()

        ids = get_context(contexts=self.contexts,
                          tickers=tickers,
                          date=timestamp,
                          max_len=self.max_len
                          )

        alloc = torch.tensor(alloc, dtype=torch.float32).view(1, 1).cuda()
        ids = torch.tensor(ids, dtype=torch.int32).view(1, self.max_len).cuda()
        state = torch.tensor(state, dtype=torch.float32)

        action, state = self.get_policy(x=(alloc, ids), state=state)
        return action, state

    def answer_requests(self):
        """
        Thread to answer actor requests from queue_request and return_episode.
        Loops through with a time gap of 0.0001 sec
        """

        while True:
            time.sleep(0.0001)

            with self.lock:

                # clear self.future2 (store episodes)
                if self.await_rpc:
                    self.await_rpc = False

                    future = self.future2
                    self.future2 = Future()
                    future.set_result(None)

                # clear self.future1 (answer requests)
                if self.pending_rpc is not None:
                    action, state = self.get_action(*self.pending_rpc)
                    self.pending_rpc = None

                    future = self.future1
                    self.future1 = Future()
                    future.set_result((action, state))

    def prepare_data(self):
        """
        Thread to prepare batch for update, batch_queue is filled by ReplayBuffer
        Loops through with a time gap of 0.1 sec
        """

        while True:
            time.sleep(0.1)

            if not self.batch_queue.empty() and len(self.batch_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batch_data.append(data)

    def run(self):
        """
        Main training loop. Start ReplayBuffer threads, answer_requests thread,
        and prepare_data thread. Then starts training
        """
        self.replay_buffer.start_threads()

        inference_thread = threading.Thread(target=self.answer_requests, daemon=True)
        inference_thread.start()

        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()

        time.sleep(2)
        while True:
            time.sleep(1)

            while not self.batch_data:
                time.sleep(0.1)
            block = self.batch_data.pop(0)

            self.update(allocs=block.allocs,
                        ids=block.ids,
                        actions=block.actions,
                        rewards=block.rewards,
                        bert_targets=block.bert_targets,
                        states=block.states,
                        idxs=block.idxs
                        )

    def update(self, allocs, ids, actions, rewards, bert_targets, states, idxs):
        """
        An update step. Performs a training step, update new recurrent states,
        soft update target model and transfer weights to eval model
        """
        loss, bert_loss, new_states = self.train_step(allocs=allocs.cuda(),
                                                      ids=ids.cuda(),
                                                      actions=actions.cuda(),
                                                      rewards=rewards.cuda(),
                                                      bert_targets=bert_targets.cuda(),
                                                      states=states.cuda()
                                                      )

        # Trick Two: “Delayed” Policy Updates.
        if self.update_steps % 2 == 0:
            policy_loss = self.update_policy(allocs=allocs.cuda(),
                                             ids=ids.cuda(),
                                             states=states.cuda()
                                             )
            self.soft_update(self.target_policy, self.policy, self.tau)

            with self.lock_model:
                self.hard_update(self.eval_policy, self.policy)

        # update new states to buffer
        self.priority_queue.put((idxs, new_states, loss, bert_loss, self.epsilon))

        # soft update target model
        self.soft_update(self.target_model1, self.model1, self.tau)
        self.soft_update(self.target_model2, self.model2, self.tau)

        return loss, bert_loss

    def update_policy(self, allocs, ids, states):
        self.policy.zero_grad()
        self.model1.zero_grad()

        cstate = states.detach()
        pstate = states.detach()
        for t in range(self.burnin_len):
            pstate = self.policy.state_forward(x=ids[t], state=pstate)
            cstate = self.model1.state_forward(x=ids[t], state=cstate)

        total_loss = 0.
        for t in range(self.burnin_len, self.rollout_len):
            action, pstate = self.policy(x=(allocs[t], ids[t]),
                                         state=pstate)
            (loss, _), _, cstate = self.model1(x=(allocs[t], ids[t]),
                                               p=action,
                                               state=cstate,
                                               n_tau=self.n_tau)
            total_loss -= loss.mean()

        total_loss.backward()
        self.policy_optimizer.step()

        return total_loss

    def train_step(self, allocs, ids, actions, rewards, bert_targets, states):
        """
        Accumulate gradients to increase batch size
        Gradients are cached for n_accumulate steps before optimizer.step()

        Parameters:
        allocs (Tensor[block_len+n_step, batch_size*n_accumulate, 1]): allocation values
        ids (Tensor[block_len+n_step, batch_size*n_accumulate, max_len]): tokens
        actions (Tensor[block_len+n_step, batch_size*n_accumulate, 1, 1]): recorded actions
        rewards (Tensor[block_len, batch_size*n_accumulate, 1]): recorded rewards
        bert_targets (Tensor[block_len+n_step, batch_size*n_accumulate, 1]): bert targets
        states (Tensor[batch_size*n_accumulate, state_len, d_model]): recorded recurrent states

        Returns:
        loss (float): Loss of critic model
        bert_loss (float): Loss of bert masked language modeling
        new_states (float): Generated new states with new weights during training

        """

        loss, bert_loss = 0, 0
        new_states = []
        grads1 = [torch.zeros(x.shape).cuda() for x in self.model1.parameters()]
        grads2 = [torch.zeros(x.shape).cuda() for x in self.model2.parameters()]

        for i in range(self.n_accumulate):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            loss_, bert_loss_, new_states_ = self.get_gradients(allocs=allocs[:, start:end, :].detach(),
                                                                ids=ids[:, start:end, :].detach(),
                                                                actions=actions[:, start:end, :].detach(),
                                                                rewards=rewards[:, start:end, :].detach(),
                                                                bert_targets=bert_targets[:, start:end, :].detach(),
                                                                states=states[:, start:end, :, :].detach()
                                                                )
            loss += loss_
            bert_loss += bert_loss_
            new_states.append(new_states_)

            for x, grad in zip(self.model1.parameters(), grads1):
                if x.grad is not None:
                    grad += x.grad
            for x, grad in zip(self.model2.parameters(), grads2):
                if x.grad is not None:
                    grad += x.grad

        for x, grad in zip(self.model1.parameters(), grads1):
            x.grad = grad / self.n_accumulate
        for x, grad in zip(self.model2.parameters(), grads2):
            x.grad = grad / self.n_accumulate

        torch.nn.utils.clip_grad_norm_(self.model1.parameters(), 0.1)
        self.optimizer1.step()
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 0.1)
        self.optimizer2.step()

        loss /= self.n_accumulate
        bert_loss /= self.n_accumulate

        new_states = np.stack(new_states)
        new_states = new_states.reshape((new_states.shape[0] * new_states.shape[1],) + new_states.shape[2:])

        return loss, bert_loss, new_states

    def get_gradients(self, allocs, ids, actions, rewards, bert_targets, states):
        """
        Memory-Efficient BPTT see https://arxiv.org/abs/1606.03401
        Overcoming memory limitations by caching gradients and chunking BPTT
        into different backpropagaton steps

        Parameters:
        allocs (Tensor[block_len+n_step, batch_size*n_accumulate, 1]): allocation values
        ids (Tensor[block_len+n_step, batch_size*n_accumulate, max_len]): tokens
        actions (Tensor[block_len+n_step, batch_size*n_accumulate, 1, 1]): recorded actions
        rewards (Tensor[block_len, batch_size*n_accumulate, 1]): recorded rewards
        bert_targets (Tensor[block_len+n_step, batch_size*n_accumulate, 1]): bert targets
        states (Tensor[batch_size*n_accumulate, state_len, d_model]): recorded recurrent states

        Returns:
        loss (float): Loss of critic model
        bert_loss (float): Loss of bert masked language modeling
        new_states (float): Generated new states with new weights during training

        """

        # create critic targets and new states
        with torch.no_grad():
            state1 = states.detach()
            state2 = states.detach()
            pstate = states.detach()

            new_states1 = []
            new_states2 = []
            for t in range(self.burnin_len+self.n_step):
                new_states1.append(state1.detach())
                new_states2.append(state2.detach())

                pstate = self.target_model1.module.state_forward(
                    ids=ids[t],
                    state=pstate
                )
                state1 = self.target_model1.module.state_forward(
                    ids=ids[t],
                    state=state1
                )
                state2 = self.target_model2.module.state_forward(
                    ids=ids[t],
                    state=state2,
                )

            next_q_values = []
            for t in range(self.burnin_len+self.n_step, self.block_len+self.n_step):
                new_states1.append(state1.detach())
                new_states2.append(state2.detach())

                # Trick Three: Target Policy Smoothing.
                target_action, pstate = self.target_policy(x=(allocs[t], ids[t]), state=pstate)
                noise = (torch.randn_like(target_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                target_action = (target_action + noise).clamp(-1, 1)

                (next_q_values1_, _), _, state1 = self.target_model1.module.forward(
                    x=(allocs[t], ids[t]),
                    p=target_action,
                    state=state1,
                    n_tau=self.n_tau
                )
                (next_q_values2_, _), _, state2 = self.target_model2.module.forward(
                    x=(allocs[t], ids[t]),
                    p=target_action,
                    state=state2,
                    n_tau=self.n_tau
                )

                assert next_q_values1_.shape == (self.batch_size, self.n_tau)
                assert next_q_values2_.shape == (self.batch_size, self.n_tau)

                # Trick One: Clipped Double-Q Learning.

                # get the minimum of two targets
                next_q_values_ = torch.stack([next_q_values1_, next_q_values2_], dim=1)
                idx = torch.argmin(next_q_values_.mean(-1), dim=1)

                # get index of two
                next_q_values_ = next_q_values_[torch.arange(next_q_values_.size(0)), idx]
                assert next_q_values_.shape == (self.batch_size, self.n_tau)

                next_q_values.append(next_q_values_)

            assert self.gamma == 0.99

            targets = rewards[self.burnin_len:] + self.gamma * torch.stack(next_q_values)
            targets = targets.view(self.rollout_len, self.batch_size, 1, self.n_tau)

        self.model1.zero_grad()
        self.model2.zero_grad()

        save_grad = torch.zeros(self.batch_size, self.state_len, self.d_model).cuda()
        loss, bert_loss = 0, 0

        intervals = list(range(self.burnin_len, self.block_len))
        for ckpt in reversed(intervals):
            ckpt_state1 = new_states1[ckpt].detach()
            ckpt_state2 = new_states2[ckpt].detach()

            assert ckpt_state1.grad is None
            ckpt_state1.requires_grad = True

            (expected1, bert_expected1), taus1, state1 = self.model1.module.forward(
                x=(allocs[ckpt], ids[ckpt]),
                p=actions[ckpt],
                state=ckpt_state1,
                n_tau=self.n_tau
            )
            (expected2, bert_expected2), taus2, state2 = self.model2.module.forward(
                x=(allocs[ckpt], ids[ckpt]),
                p=actions[ckpt],
                state=ckpt_state2,
                n_tau=self.n_tau
            )

            target = targets[ckpt-self.burnin_len].view(self.batch_size, 1, self.n_tau)
            expected1 = expected1.view(self.batch_size, self.n_tau, 1)
            taus1 = taus1.view(self.batch_size, self.n_tau, 1)
            expected2 = expected2.view(self.batch_size, self.n_tau, 1)
            taus2 = taus2.view(self.batch_size, self.n_tau, 1)

            # bert_target = bert_targets[ckpt]
            # bert_expected = bert_expected.transpose(1, 2)

            loss1_ = self.quantile_loss(expected1, target, taus1)
            loss1_.backward()

            loss2_ = self.quantile_loss(expected2, target, taus2)
            loss2_.backward()

            loss_ = loss1_ + loss2_

            # torch.autograd.backward([loss_, state], [None, save_grad])
            bert_loss_ = torch.tensor(0.)

            # loss_, bert_loss_, ckpt_state = self.get_gradients_step(expected=expected,
            #                                                         target=target,
            #                                                         taus=taus,
            #                                                         bert_expected=bert_expected,
            #                                                         bert_target=bert_target,
            #                                                         state=state,
            #                                                         save_grad=save_grad,
            #                                                         ckpt_state=ckpt_state
            #                                                         )

            if ckpt != self.burnin_len:
                assert ckpt_state1.grad is not None
                save_grad1 = ckpt_state1.grad
                save_grad2 = ckpt_state2.grad

            loss += loss_
            bert_loss += bert_loss_

        loss /= self.rollout_len
        bert_loss /= self.rollout_len

        for x in self.model1.parameters():
            if x.grad is not None:
                x.grad.data.mul_(1/self.rollout_len)
        for x in self.model2.parameters():
            if x.grad is not None:
                x.grad.data.mul_(1/self.rollout_len)

        loss = loss.detach().cpu().numpy().item()
        bert_loss = bert_loss.detach().cpu().numpy().item()

        # shape has to be [batch_size, timesteps, ...]
        new_states1 = torch.stack(new_states1).transpose(1, 2).transpose(0, 1).unsqueeze(3).detach().cpu().numpy()

        return loss, bert_loss, new_states1

    def get_gradients_step(self, expected, target, taus, bert_expected, bert_target, state, save_grad, ckpt_state):
        """
        Used concepts from PCGrad see https://arxiv.org/pdf/2001.06782.pdf
        The idea is to do multi-task learning by treating gradients as vectors
        and projecting then onto each other, then removing conflicting directions
        The idea is to train bert masked language modeling and critic for rl
        at the same time to accelerate training by overcoming the small signal
        from rl rewards with signal from masked language modeling

        Parameters:
        expected (Tensor[batch_size, n_tau, 1]): expected critic values
        target (Tensor[batch_size, 1, n_tau]): target critic values
        taus (Tensor[batch_size, n_tau, 1]): taus
        bert_expected (Tensor[batch_size, max_len, vocab_size]): expected bert values
        bert_target (Tensor[batch_size, max_len]): target bert values
        state (Tensor[batch_size, state_len, d_model]): recurrent state to cache gradients
        save_grad (Tensor[batch_size, state_len, d_model]): saved state gradients from next step
        ckpt_state (Tensor[batch_size, state_len, d_model]): next recurrent state associated with save_grad

        Returns:
        loss(Tensor[]): Critic Loss
        bert_loss(Tensor[]): Bert masked language modeling loss
        ckpt_state(Tensor[batch_size, state_len, d_model]): tensor of recurrent state with cached gradients
        """
        assert target.shape == (self.batch_size, 1, self.n_tau)
        assert expected.shape == (self.batch_size, self.n_tau, 1)
        assert taus.shape == (self.batch_size, self.n_tau, 1)
        assert bert_expected.shape == (self.batch_size, 30522, self.max_len)
        assert bert_target.shape == (self.batch_size, self.max_len)
        assert state.shape == (self.batch_size, self.state_len, self.d_model)
        assert save_grad.shape == (self.batch_size, self.state_len, self.d_model)

        grads = [x.grad for x in self.model.parameters()]

        # get critic grads
        self.model.zero_grad()
        loss = self.quantile_loss(expected, target, taus)
        torch.autograd.backward([loss, state], [None, save_grad], retain_graph=True)
        critic_grads = [x.grad for x in self.model.parameters()]
        critic_state_grad = ckpt_state.grad
        ckpt_state.grad = None
        assert critic_state_grad is not None

        # get bert grads
        self.model.zero_grad()
        bert_loss = self.bert_loss(bert_expected, bert_target)
        torch.autograd.backward([bert_loss, state], [None, save_grad])
        bert_grads = [x.grad for x in self.model.parameters()]
        bert_state_grad = ckpt_state.grad
        ckpt_state.grad = None
        assert bert_state_grad is not None

        # project state gradients
        critic_state_grad, bert_state_grad = self.proj_grads(critic_state_grad, bert_state_grad)
        ckpt_state.grad = (critic_state_grad + bert_state_grad) / 2

        # project gradients
        for i in range(len(grads)):
            if critic_grads[i] is not None and bert_grads[i] is not None:
                critic_grads[i], bert_grads[i] = self.proj_grads(critic_grads[i], bert_grads[i])

        # put gradients back into model
        self.model.zero_grad()
        for i, x in enumerate(self.model.parameters()):

            if grads[i] is None:
                grads[i] = torch.zeros(x.shape).cuda()
            if critic_grads[i] is None:
                critic_grads[i] = torch.zeros(x.shape).cuda()
            if bert_grads[i] is None:
                bert_grads[i] = torch.zeros(x.shape).cuda()

            x.grad = grads[i] + ((critic_grads[i] + bert_grads[i]) / 2)

        assert ckpt_state.grad is not None
        return loss, bert_loss, ckpt_state

    @staticmethod
    def proj_grads(grad1, grad2):
        """
        Project gradients function from PCGrad see https://arxiv.org/pdf/2001.06782.pdf
        Projecting u onto v and removing it if it's in the opposite direction
        See equation https://www.youtube.com/watch?v=m4PZBk8Zi8w

        Parameters:
        grad1 (Tensor[batch_size, ...]): first gradient
        grad2 (Tensor[batch_size, ...]): second gradient

        Returns:
        grad1 (Tensor[batch_size, ...]): projected first gradient
        grad2 (Tensor[batch_size, ...]): projected second gradient
        """
        grad1_ = grad1.detach()

        # project grad1
        proj_direction = torch.sum(grad1 * grad2) / (torch.sum(grad2 * grad2) + torch.tensor(1e-12))
        grad1 = grad1 - torch.min(proj_direction, torch.tensor(0.)) * grad2
        assert not torch.isnan(grad1).any()

        # project grad2
        proj_direction = torch.sum(grad2 * grad1_) / (torch.sum(grad1_ * grad1_) + torch.tensor(1e-12))
        grad2 = grad2 - torch.min(proj_direction, torch.tensor(0.)) * grad1_
        assert not torch.isnan(grad2).any()

        return grad1, grad2

    def quantile_loss(self, expected, target, taus):
        """
        See IQN: https://arxiv.org/pdf/1806.06923.pdf
        Loss equation is Page 5 Eqn (3)
        Training a distribution by viewing the target as a distribution and approximating it

        Parameters:
        expected (Tensor[batch_size, n_tau, 1])
        target (Tensor[batch_size, 1, n_tau])
        taus (Tensor[batch_size, n_tau, 1])

        Returns:
        loss (Tensor[]): quantile loss

        """
        assert not taus.requires_grad

        assert expected.shape == (self.batch_size, self.n_tau, 1)
        assert target.shape == (self.batch_size, 1, self.n_tau)
        assert taus.shape == (self.batch_size, self.n_tau, 1)

        td_error = target - expected
        huber_loss = torch.where(td_error.abs() <= 1, 0.5 * td_error.pow(2), td_error.abs() - 0.5)
        quantile_loss = abs(taus - (td_error.detach() < 0).float()) * huber_loss

        critic_loss = quantile_loss.sum(dim=1).mean(dim=1)
        critic_loss = critic_loss.mean()

        return critic_loss

    def bert_loss(self, expected, target):
        """
        Standard bert masked language modeling loss.

        Parameters:
        expected (Tensor[batch_size, vocab_size, max_len]): expected values
        target (Tensor[batch_size, max_len]): target values

        Returns:
        loss (Tensor[]): bert loss

        """
        assert expected.shape == (self.batch_size, self.vocab_size, self.max_len)
        assert target.shape == (self.batch_size, self.max_len)

        return self.nll_loss(expected, target)

    def load(self, name="checkpoint"):
        """Load weights from saved directory"""
        state_dict = torch.load(f"saved/{name}")

        # self.model.load_state_dict(state_dict)
        # self.target_model.load_state_dict(state_dict)
        # self.eval_model.load_state_dict(state_dict)

    def save(self, name="checkpoint"):
        """Save weights to saved directory"""
        torch.save(self.model.state_dict(), f"saved/{name}")

    @staticmethod
    def soft_update(target, source, tau):
        """
        Soft weight updates: target slowly track the weights of source with constant tau
        See DDPG paper page 4: https://arxiv.org/pdf/1509.02971.pdf

        Parameters:
        target (nn.Module): target model
        source (nn.Module): source model
        tau (float): soft update constant
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        """
        Copy weights from source to target

        Parameters:
        target (nn.Module): target model
        source (nn.Module): source model
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
