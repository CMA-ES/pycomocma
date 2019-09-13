%'MYPB','DTLZ1','DTLZ2','DTLZ3','DOUBLESPHERE','ELLInonrotated','ELLI','ELLI2','CIGTABnonrotated','CIGTAB','CIGTAB2','FON','ZDT1';
% ProblemName='DOUBLESPHERE';
ProblemName='MYPB';
opts = COMOCMAES();
opts.nPop = 15;
nObj = 2;
nVar = 20;
opts.tolx=1e-4;
sigma0 = 0.2;
opts.elitist=0;

%opts.number_asks =opts.nPop;%'1 % pour sequentiel'
%opts.maxiter=fix(10000/opts.number_asks);
%opts.number_asks = 1;
opts.maxiter = 500;
xstart = rand(1, nVar);

opts.logger = 10; % to write data every opts`.logger` iterations
opts.okresume = 0; % option for resume: `1` means we save the optimization
opts.resumefile='';
opts.verb_disp=50;
opts.abscissa = 1;

% adapt bounds
if strcmp(ProblemName,'MYPB')
    LB=-5*ones(1,nVar);% lower bound
    HB=-LB;% upper bound
else
    LB=0*ones(1,nVar);% lower bound
    HB=1*ones(1,nVar);% upper bound
end
 LB=LB';
 HB=HB';
 
bounds=[LB,HB,ones(size(LB)),ones(size(LB))];
opts.bounds =[LB,HB];% [0.2, 0.9];

% after each iteration (for difficult problems), and for `0` we skip the saving
%opts.display = 1;
% opts.verb_disp = 100;
if strcmp(ProblemName,'DOUBLESPHERE')
    opts.refpoint    = [1.1; 1.1];% [0.8;0.6];
    problem=@(x) bi_sphere(x);
else
    if nObj==2
        % pour MYPB a 2 objectifs
        opts.refpoint    =[30; 60];
    elseif nObj==3
        % pour MYPB a 3 objectifs
        opts.refpoint    =[20;30;-100];
    end
    CovFobj=diag(ones(nObj,1)).*0.0;
    %cheikh
    penaltyfactor = [0, 0];
    ContraintesFC = [0, 0];
    
    %end cheikh
    hfig=NaN;
%    ContraintesFC=[Inf;Inf];
    % mise � jour ru refpoint
 %   x=repmat(LB,1,1000)+(repmat(HB,1,1000)-repmat(LB,1,1000)).*rand(nVar,1000);
  %  F=problem(x');
   % F=F(:,1:nObj)';
   % opts.refpoint=(max(F,[],2)); %commenter refpoint pour avoir un
   % refpoint fixe et pouvoir comparer number_asks = 1 et number_asks = "all"
    problem=@(x)MyFunCmaesFHUprob(x,nObj, penaltyfactor, bounds, ContraintesFC, ProblemName, CovFobj, hfig);
end
T0=clock;
[paretoFront, paretoSet, out] = COMOCMAES(problem, nObj, xstart, sigma0, opts);
fprintf(' optim time lapse= %f min\n',etime(clock,T0)/60);

