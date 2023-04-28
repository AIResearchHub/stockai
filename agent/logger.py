import datetime
import time


class Logger:
    """
`   Prints and logs data, parameters are modified directly by buffer threads
    """

    def __init__(self):
        self.datetime = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
        self.file = open(f"logs/{self.datetime}", "w")

        self.total_frames = 0
        self.total_updates = 0
        self.loss = 0
        self.bert_loss = 0
        self.reward = 0

        self.start = time.time()

    def print(self):
        elapsed_time = time.time() - self.start

        if self.loss != 0 or self.bert_loss != 0:
            self.file.write('{}, {}, {}, {}, {}, {} \n'.format(elapsed_time,
                                                               self.total_updates,
                                                               self.total_frames,
                                                               self.loss,
                                                               self.bert_loss,
                                                               self.reward))
            self.file.flush()

        print('Elapsed: {:>8.4f}  Updates: {:>8}  Frames: {:>8} Loss: {:>10.8f} BertLoss: {:>10.8f} Reward: {:>10.4f}'
              .format(elapsed_time,
                      self.total_updates,
                      self.total_frames,
                      self.loss,
                      self.bert_loss,
                      self.reward
                      ),
              flush=True)

