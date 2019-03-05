import torch, math, tqdm, copy, random
import torch.utils.data

from MovingAverage import MovingAverage

class Model(torch.nn.Module):

    def __init__(self, D):
        super().__init__()
        self.net = torch.nn.Sequential(

            torch.nn.Linear(D, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1)

        )

    def forward(self, X):
        return self.net(X).squeeze(-1)

class EvolutionaryModel:

    def __init__(self, D, disable):
        self.pool = [EvolutionaryUnit(D) for i in range(8)]
        self.cycle = 4 if not disable else 9999
        self.mutation_rate = 2 if not disable else 9999
        self.i = 0
        self.max_adult_pop = 16
        self.max_pop = 32
        self.softmax = torch.nn.Softmax(dim=0)

    def select_best(self):
        return self.pool[0]

    def to(self, device):
        for unit in self.pool:
            unit.set_device(device)
        return self

    def do_cycle(self, X, Y, X_test, Y_test):

        for unit in self.pool:
            unit.fit(X, Y, X_test, Y_test)

        self.i += 1
        if self.i % self.cycle == self.mutation_rate:
            self.pool.extend([unit.mutate() for unit in self.pool])
            random.shuffle(self.pool)
            self.pool = self.pool[:self.max_pop]
        if self.i % self.cycle == 0:
            self.populate()

        print("Population size: %d" % len(self.pool))
        best = min(self.pool)
        print("Cycle %d best score: %.3f (%.3f training)" % (self.i, best.get_score(), best.get_train_score()))

    def populate(self):
        sorted_units = sorted(self.pool)[:self.max_adult_pop]
        scores = [u.get_score() for u in sorted_units]
        probs = self.softmax(-torch.Tensor(scores))
        self.pool = sorted_units + self.mate(list(zip(sorted_units, probs)))
        random.shuffle(self.pool)
        self.pool = self.pool[:self.max_pop]

    def mate(self, unitprobs):
        children = []
        for u1, p1 in unitprobs:
            for u2, p2 in unitprobs:
                if u1 is not u2:
                    if random.random() < max(p1, p2):
                        children.extend(u1.share_abilities(u2))
        print("Offspring size: %d" % len(children))
        return children

class EvolutionaryUnit(Model):

    def __init__(self, D):
        super().__init__(D)
        self.target_modules = [torch.nn.Linear]
        self.gain_rate = 0.05
        self.loss_rate = 0.20

        self.epochs = 100
        self.batch = 8
        self.score = MovingAverage(momentum=0.90)
        self.train_score = MovingAverage(momentum=0.9)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def __lt__(self, other):
        return self.get_score() < other.get_score()

    def fit(self, X, Y, X_test, Y_test):

        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch, shuffle=True)

        lossf = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.parameters())
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60])

        for epoch in tqdm.tqdm(range(self.epochs), ncols=80):

            self.train()
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                yh = self(x)
                loss = lossf(yh, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                self.train_score.update(self.calc_score(yh, y))

            sched.step()

        with torch.no_grad():
            self.eval()
            yh_test = self(X_test.to(self.device))
            self.score.update(self.calc_score(yh_test, Y_test.to(self.device)))

    def get_score(self):
        return self.score.peek() # lower the better

    def get_train_score(self):
        return self.train_score.peek()

    def mutate(self):
        out = copy.deepcopy(self)
        out.apply(self.recurse_apply(self.gain_ability))
        out.apply(self.recurse_apply(self.lose_ability))
        return out

    def share_abilities(self, other):
        n1 = copy.deepcopy(self)
        n2 = copy.deepcopy(other)
        for m1, m2 in zip(n1.net, n2.net):
            self.exchange(m1, m2)
        return n1, n2

    # === PRIVATE ===

    def exchange(self, m1, m2):
        if type(m1) in self.target_modules:
            p = 0.5
            i = torch.rand_like(m1.weight.data) < p
            m1.weight.data[i] = m2.weight.data[i]
            m2.weight.data[1-i] = m1.weight.data[1-i]

    def calc_score(self, yh, y):
        return (yh-y).abs().sum().item()

    def recurse_apply(self, f):
        return lambda m: f(m) if type(m) in self.target_modules else None

    def gain_ability(self, m):
        i, v = self.get_new_weights(m, self.gain_rate)
        torch.nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        m.weight.data[i] = v[i]

    def lose_ability(self, m):
        i, v = self.get_new_weights(m, self.loss_rate)
        m.weight.data[i] = v[i]

    def get_new_weights(self, m, rate):
        i = (torch.rand_like(m.weight.data) < rate)
        v = torch.zeros_like(m.weight.data)
        return i, v

