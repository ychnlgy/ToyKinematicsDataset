import torch, tqdm
import torch.utils.data

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot



def train(X, Y, X_test, Y_test, model):

    dataloader = create_dataloader(X, Y, 8)

    lossf = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60])

    EPOCHS = 100
    for epoch in tqdm.tqdm(range(EPOCHS)):

        model.train()
        for x, y in dataloader:
            yh = model(x)
            loss = lossf(yh, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        sched.step()

    with torch.no_grad():
        model.eval()
        yh_test = model(X_test)
        yh = model(X)
        print("Train/test difference: %.3f/%.3f" % (score(yh, Y), score(yh_test, Y_test)))

def visualize(model):

    model.to("cpu")
    
    a = torch.zeros(100) + 0.5
    v = torch.zeros(100) + 0.5
    t = torch.linspace(0, 1, 100)
    d = t*v + 0.5*a*t**2

    xp = torch.cat([t.unsqueeze(1), v.unsqueeze(1), a.unsqueeze(1), torch.rand(100, X.size(1)-3)], dim=1)
    dh = model(xp)

    pyplot.plot(t.numpy(), d.numpy(), label="Ground truth")
    pyplot.plot(t.numpy(), dh.numpy(), label="Predicted trajectory")
    pyplot.legend()
    
    pyplot.savefig("results.png")
