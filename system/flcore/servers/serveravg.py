import time
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.rs_epsilon = []
        self.Budget = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        sigma_schedule = getattr(self.args, 'sigma_schedule', None)
        T = max(self.global_rounds, 1)

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # Per-round sigma schedule: linear decay from sigma_start to sigma_end
            if sigma_schedule is not None:
                t_frac = i / T
                for client in self.selected_clients:
                    s_start, s_end = sigma_schedule[client.id]
                    client.dp_sigma = s_start + (s_end - s_start) * t_frac

            for client in self.selected_clients:
                client.train()

            if getattr(self.args, 'privacy', False):
                epsilons = [c.current_epsilon for c in self.selected_clients]
                avg_eps = sum(epsilons) / len(epsilons) if epsilons else 0.0
                self.rs_epsilon.append(avg_eps)

            self.write_progress(i)
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                    acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy:", max(self.rs_test_acc))
        print("\nAverage time cost per round:",
              sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()
