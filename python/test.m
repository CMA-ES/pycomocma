opts = COMOCMAES;
opts.nPop = 4;
nObj = 2;
nVar = 10;
xstart = ones(1, nVar);
sigma0 = 0.2;
opts.bounds = [0.5, inf];
[paretoFront, paretoSet, out] = COMOCMAES('bi_sphere', nObj, xstart, sigma0);
