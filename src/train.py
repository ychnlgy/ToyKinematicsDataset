import torch, tqdm
import torch.utils.data

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

def score(yh, y):
    return (yh-y).abs().sum().item()

def train(X, Y, X_test, Y_test, model):

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    lossf = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60])

    EPOCHS = 200
    for epoch in tqdm.tqdm(range(EPOCHS)):

        model.train()
        for x, y in dataloader:
            yh = model(x)
            loss = lossf(yh, y)# + model.penalty()
            optim.zero_grad()
            loss.backward()
            optim.step()

            #model.net[0].weight.data[:,3:]=0

        sched.step()

    with torch.no_grad():
        model.eval()
        yh_test = model(X_test)
        yh = model(X)
        print("Train/test difference: %.3f/%.3f" % (score(yh, Y), score(yh_test, Y_test)))

        visualize(model, "results.png", X.size(1))

def visualize(model, outf, D):

    with torch.no_grad():

        model.to("cpu")

        N = 1000
        a = torch.zeros(N) + 0.5
        v = torch.zeros(N) + 0
        t = torch.linspace(-2, 10, N)
        d = t*v + 0.5*a*t**2

        xp = torch.cat([t.unsqueeze(1), v.unsqueeze(1), a.unsqueeze(1), torch.rand(N, D-3)], dim=1)
        dh = model(xp)

        pyplot.plot(t.numpy(), d.numpy(), label="Ground truth")
        pyplot.plot(t.numpy(), dh.numpy(), label="Predicted trajectory")
        pyplot.legend()

        pyplot.savefig(outf)
    print("Saved visualization to: %s." % outf)
