# Compare this against what I already have and decide about replacement

def common_metrics(self, metric_type):
    num_samples = []
    tot_loss = []
    IDs = []
    curr_live_loss = []
    curr_live_num_samples = []
    curr_live_IDs = []
    prev_live_loss = []
    prev_live_num_samples = []
    prev_live_IDs = []
    unseen_live_loss = []
    unseen_live_num_samples = []
    unseen_live_IDs = []

    for i, c in enumerate(self.clients):
        if (self.sequential and c.ID in self.static_client_IDs):
            if metric_type == 'train':
                tl, ns = c.train_metrics(model_obj=self.global_model)
            elif metric_type == 'test':
                tl, ns = c.test_metrics(model_obj=self.global_model)
        else:
            if metric_type == 'train':
                tl, ns = c.train_metrics()
            elif metric_type == 'test':
                tl, ns = c.test_metrics()

        if (not self.sequential) or (self.sequential and c.ID in self.static_client_IDs):
            tot_loss.append(tl * 1.0)
            num_samples.append(ns)
            IDs.append(c.ID)
        elif self.sequential and c.ID in [lc.ID for lc in self.live_clients]:
            curr_live_loss.append(tl * 1.0)
            curr_live_num_samples.append(ns)
            curr_live_IDs.append(c.ID)
        elif self.sequential and c.ID in self.prev_live_client_IDs:
            prev_live_loss.append(tl * 1.0)
            prev_live_num_samples.append(ns)
            prev_live_IDs.append(c.ID)
        elif self.sequential and c.ID in self.unseen_live_client_IDs:
            unseen_live_loss.append(tl * 1.0)
            unseen_live_num_samples.append(ns)
            unseen_live_IDs.append(c.ID)
        elif self.sequential and c.ID in self.live_client_IDs_queue:
            pass
        elif self.sequential:
            raise ValueError("This isn't supposed to run...")
        else:
            raise ValueError("This isn't supposed to run...")

    if self.sequential:
        seq_metrics = [curr_live_loss, curr_live_num_samples, curr_live_IDs, prev_live_loss, prev_live_num_samples,
                       prev_live_IDs, unseen_live_loss, unseen_live_num_samples, unseen_live_IDs]
    else:
        seq_metrics = None

    return IDs, num_samples, tot_loss, seq_metrics


def test_metrics(self):
    if self.eval_new_clients and self.num_new_clients > 0:
        self.fine_tuning_new_clients()
        return self.test_metrics_new_clients()

    return self.common_metrics('test')


def train_metrics(self):
    self.global_round += 1

    if self.eval_new_clients and self.num_new_clients > 0:
        print("KAI: Returned early for some reason, idk what this code is doing")
        return [0], [1], [0]

    return self.common_metrics('train')