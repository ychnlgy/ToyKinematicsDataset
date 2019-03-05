import dataset, model, train

if __name__ == "__main__":

    device = ["cpu", "cuda"][torch.cuda.is_available()]
    print("Using device: %s" % device)

    N = 512
    D = 32

    data = dataset.create(N, D)
    test = dataset.test(N, D)

    net = model.EvolutionaryModel(D).to(device)

    for i in range(100):
        net.do_cycle(*data, *test)
    #train.train(*data, *test, net)
