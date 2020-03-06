opts = COMOCMAES();
opts.nPop = 5;
nObj = 2;
nVar = 20;
xstart = ones(1, nVar);
sigma0 = 0.2;
opts.bounds = [0.2, 0.9];
%opts.maxiter = 400;
opts.number_asks = 1;
opts.logger = 1; % to write data
opts.okresume = 0; % option for resume: `1` means we save the optimization
% after each iteration (for difficult problems), and for `0` we skip the saving
%opts.display = 1;
% opts.verb_disp = 100;
[paretoFront, paretoSet, out] = COMOCMAES('bisphere', nObj, xstart, sigma0, opts);
