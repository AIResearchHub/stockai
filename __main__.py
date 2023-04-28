

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import portpicker
import os
import time

from agent import Learner


def run_worker(rank,
               tickers,
               mock_data,
               buffer_size,
               batch_size,
               n_accumulate,
               vocab_size,
               n_layers,
               d_model,
               n_head,
               state_len,
               burnin_len,
               rollout_len,
               n_step,
               n_cos,
               n_p,
               n_tau
               ):
    """
    Workers:
        - Learner.run()
        - Actor.run()

    Threads:
        - Learner.answer_requests(time.sleep(0.0001))
        - Learner.prepare_data(time.sleep(0.1))
        - ReplayBuffer.add_data()
        - ReplayBuffer.prepare_data(time.sleep(0.1))
        - ReplayBuffer.update_data(time.sleep(0.1))
        - ReplayBuffer.log_data(time.sleep(10))

    """

    if rank == 0:
        rpc.init_rpc("learner", rank=rank, world_size=2)

        learner_rref = rpc.remote(
            "learner",
            Learner,
            args=(buffer_size,
                  batch_size,
                  n_accumulate,
                  tickers,
                  mock_data,
                  vocab_size,
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
                  ),
            timeout=0
        )
        learner_rref.remote().run()

        while True:
            time.sleep(1)

    else:
        rpc.init_rpc(f"actor", rank=rank, world_size=2)

    rpc.shutdown()


def main(tickers,
         mock_data,
         buffer_size,
         batch_size,
         n_accumulate,
         vocab_size,
         n_layers,
         d_model,
         n_head,
         state_len,
         burnin_len,
         rollout_len,
         n_step,
         n_cos,
         n_p,
         n_tau
         ):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(portpicker.pick_unused_port())

    mp.spawn(
        run_worker,
        args=(tickers,
              mock_data,
              buffer_size,
              batch_size,
              n_accumulate,
              vocab_size,
              n_layers,
              d_model,
              n_head,
              state_len,
              burnin_len,
              rollout_len,
              n_step,
              n_cos,
              n_p,
              n_tau
              ),
        nprocs=2,
        join=True
    )


if __name__ == "__main__":
    main(tickers=["AAPL", "AMZN"],
         mock_data=True,
         buffer_size=100000,
         batch_size=16,
         n_accumulate=1,
         vocab_size=30522,
         n_layers=4,
         d_model=512,
         n_head=8,
         state_len=1,
         burnin_len=10,
         rollout_len=20,
         n_step=1,
         n_cos=64,
         n_p=128,
         n_tau=64
         )

