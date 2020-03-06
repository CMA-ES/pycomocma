function F=MyFunCmaesFHUprob(x,nObj, penaltyfactor, bounds, ContraintesFC, Problem, CovFobj, hfig)
% objective function with noise handle various problems :
%'MYPB','DTLZ1','DTLZ2','DTLZ3','DOUBLESPHERE','ELLInonrotated','ELLI','ELLI2','CIGTABnonrotated','CIGTAB','CIGTAB2','FON','ZDT1';
x=x'; % pour comcmaes
[n,m]=size(x);
if nargin==7 || nObj>1
    hfig=NaN;
end
feasx = x;
%feasx = feasible(x, bounds);
%cst_bounds= penaltyterm(x, feasx, penaltyfactor(1),bounds);
if size(bounds,2)==4
    % soft and hard penalty
    bin_inf=x<feasx & repmat(bounds(:,3)==0,1,m);
    x(bin_inf)=feasx(bin_inf);
     bin_sup=x>feasx & repmat(bounds(:,4)==0,1,m);
    x(bin_sup)=feasx(bin_sup);
end
%feasx=x;
% x=feasx;
if nObj==1
    no=2;
else
    no=nObj;
end
try
    F=zeros(no,m);
    for i=1:m
        F(:,i)=feval(Problem,x(:,i),no);
    end
catch
    error([ Problem,' not in the list of predefined problems']);
end

% DF=F - repmat(ContraintesFC,[1,m]);% contraintes sur le max de chaque objectif
% % DF=[F - repmat(ContraintesFC,[1,m]);sum(F,1)-ones(1,m).*(-60)];% contraintes sur le max de chaque objectif et la somme <-60
% DF(DF<0)=0;
% f_penalty=penaltyfactor(2) .*DF;%+sum(DF2/10));
% 
% if nObj==1
%    F=sum(F); 
% end
% if any(CovFobj(:)~=0)
%     F=(F'+randn(m,nObj)*chol(CovFobj))';
% end
% 
% 
% % add xpenalty for bound constraints
% if nObj==1
%     F=sum(F,1);
% end
% cst=[cst_bounds;f_penalty];
if ishandle(hfig)
    
    f1=bounds(1,1):0.1:bounds(1,2);
    f2=bounds(2,1):0.1:bounds(2,2);
    [Y,X]=meshgrid(f2,f1);
    s=size(X);
    axes(hfig);cla;
    Fim=zeros(no,numel(X));
    xy=[X(:),Y(:)]';
    for i=1:numel(X)
        Fim(:,i)=feval(Problem,xy(:,i),no);
    end
    if nObj==1
        Fim=sum(Fim);
    end
    Fim=reshape(Fim(:),s);
    imagesc(f1,f2,Fim');set(gca,'YDir','normal');xlabel('x1');ylabel('x2');colorbar
    plot(x(1,:)',x(2,:)','.k');
    [Fmin,pos]=min(F);
    xmean=mean(x,2);
    plot(x(1,pos)',x(2,pos)','+r');
    plot(xmean(1),xmean(2),'+m');
end
% pour comocmaes lignes offsprings colonnes obj et contraintes
%F=[F',cst'];
F = F';

function feasx = feasible(x, bounds)
    % in x: one column per solution, variables in rows
    % in bounds: column1: lower bound; column2: upper bound; rows=variables
	feasx = min(max(x, repmat(bounds(:,1), 1, size(x,2))), ...
                repmat(bounds(:,2), 1, size(x,2)));


function cst_bounds = penaltyterm(x, feasx, penaltyfactor, bounds)
    % x,feasx: one column per solution, variables in rows
    dx=(x - feasx);
    penalizedf_inf=-(dx<0).*dx.*(1.001-repmat(bounds(:,3),1,size(x,2)));
    penalizedf_sup=(dx>0).*dx.*(1.001-repmat(bounds(:,3),1,size(x,2)));
    cst_bounds=[sum(penalizedf_inf);sum(penalizedf_sup)].*penaltyfactor;
%     dx(dx<0)=0;
%     if nargin==4
%         penalizedf =100* penaltyfactor .* sum(dx./repmat(diff(bounds,1,2), [1, size(x,2)]));
%     else
%         penalizedf =penaltyfactor .* sum(dx);
%     end
  
        
%% Problems
function [obj, feasx] = DTLZ1(x, nObj, penaltyfactor)
    % returns objective vectors 'obj' as well as decision vectors projected
    % onto the boundaries ('feasx')
    % for solutions outside the domain boundaries, a penalization term
    % is added.
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    bounds = [zeros(nVar,1) ones(nVar,1)];
    
    % compute unpenalized objective functions first:
    feasx = feasible(x, bounds);
    g = 100.*( (nVar-nObj+1) + ...
            sum( (feasx(nObj:end,:) - 0.5).^2 - cos(20.*pi.*(feasx(nObj:end,:) - 0.5) )));
    obj = 0.5 .* repmat((1+g), nObj, 1);
    for i=1:nObj-1
        for j = 1:nObj-i
            obj(i,:) = obj(i,:) .* feasx(j,:);
        end
    end
    for i=2:nObj
        obj(i,:) = obj(i,:) .* (1-feasx(nObj-i+1,:));
    end
    
    % add penalization term:
    for i=1:nObj
        obj(i,:) = obj(i,:) + penaltyterm(x, feasx, penaltyfactor);
    end
    
    
function [obj, feasx] = DTLZ2(x, nObj, penaltyfactor)
    % typically nVar = 9+nObj (recommended in the original publication)
    % returns objective vectors 'obj' as well as decision vectors projected
    % onto the boundaries ('feasx')
    % for solutions outside the domain boundaries, a penalization term
    % is added.
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    bounds = [zeros(nVar,1) ones(nVar,1)];
    
    % compute unpenalized objective functions first:
    feasx = feasible(x, bounds);
    g = sum( (feasx(nObj:end,:) - 0.5).^2 );
    obj = repmat((1+g), nObj, 1);
    for i=1:nObj-1
        for j = 1:nObj-i
            obj(i,:) = obj(i,:) .* cos(feasx(j,:) .* pi/2);
        end
    end
    for i=2:nObj
        obj(i,:) = obj(i,:) .* sin(feasx(nObj-i+1,:) .* pi/2);
    end
    
    % add penalization term:
    for i=1:nObj
        obj(i,:) = obj(i,:) + penaltyterm(x, feasx, penaltyfactor);
    end

        
function [obj, feasx] = DTLZ3(x, nObj, penaltyfactor)
    % typically nVar = 9+nObj (recommended in the original publication)
    % returns objective vectors 'obj' as well as decision vectors projected
    % onto the boundaries ('feasx')
    % for solutions outside the domain boundaries, a penalization term
    % is added.
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    bounds = [zeros(nVar,1) ones(nVar,1)];
    
    % compute unpenalized objective functions first:
    feasx = feasible(x, bounds);
    g = 100.*( (nVar-nObj+1) + ...
            sum( (feasx(nObj:end,:) - 0.5).^2 - cos(20.*pi.*(feasx(nObj:end,:) - 0.5) )));
    obj = repmat((1+g), nObj, 1);
    for i=1:nObj-1
        for j = 1:nObj-i
            obj(i,:) = obj(i,:) .* cos(feasx(j,:) .* pi/2);
        end
    end
    for i=2:nObj
        obj(i,:) = obj(i,:) .* sin(feasx(nObj-i+1,:) .* pi/2);
    end
    
    % add penalization term:
    for i=1:nObj
        obj(i,:) = obj(i,:) + penaltyterm(x, feasx, penaltyfactor);
    end



function [obj, feasx] = DOUBLESPHERE(x, nObj, penaltyfactor)
    % simple two-sphere function
    % note that this is an unconstrained benchmark problem
    
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem DOUBLESPHERE only works for 2 objectives');
    end
    
    obj = zeros(2, N);
	obj(1,:) = sum( x.^2 ) ./ nVar ./4;
	obj(2,:) = sum( (x-2).^2 ) ./ nVar ./4;
    feasx = x;

    
function [obj, feasx] = ELLInonrotated(x, nObj, penaltyfactor)
    % two-ellipsoid function, unrotated
    % note that this is an unconstrained benchmark problem
    % according to original MO-CMA-ES paper, this problem has 10 variables
    
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem ELLInonrotated only works for 2 objectives');
    end
    a = 10; % parameter of the function
    
    obj = zeros(2, N);
    exponent = 0:1:nVar-1;
    constA = repmat((a.^(2.*exponent./(nVar-1)))', 1, N);
    constB = repmat(a.*a.*nVar, 1, N);
	obj(1,:) = sum(constA .* x.^2 )./constB;
	obj(2,:) = sum(constA .* (x-2).^2 )./constB;
    feasx = x;


function [obj, feasx] = ELLI(x, nObj, penaltyfactor)
    % two-ellipsoid function, rotated
    % note that this is an unconstrained benchmark problem
    % according to original MO-CMA-ES paper, this problem has 10 variables
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem ELLI only works for 2 objectives');
    end
    
    % rotate coordinate system:
    B = computeRotation(nVar, 1);
    y = zeros(nVar, N);
    
    for i=1:N
        y(:,i) = B*x(:,i);
    end
    
    [obj, feasx] = ELLInonrotated(x, nObj, penaltyfactor);
    
    
function [obj, feasx] = ELLI2(x, nObj, penaltyfactor)
    % two-ellipsoid function, coordinate system rotated independently for
    % each objective function
    % note that this is an unconstrained benchmark problem
    % according to original MO-CMA-ES paper, this problem has 10 variables
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem ELLI2 only works for 2 objectives');
    end
    
    obj = zeros(2, N);
    % rotate coordinate systems:
    B = computeRotation(nVar, 1);
    y = zeros(nVar, N);
    for i=1:N
        y(:,i) = B*x(:,i);
    end
    B2 = computeRotation(nVar, 2);
    z = zeros(nVar, N);
    for i=1:N
        z(:,i) = B2*x(:,i);
    end
    
    a = 1000; % parameter of the function
    
    obj = zeros(2, N);
    exponent = 0:1:nVar-1;
	obj(1,:) = sum(repmat((a.^(2.*exponent./(nVar-1)))', 1, N) .* y.^2 )./repmat(a.*a.*nVar, 1, N);
	obj(2,:) = sum(repmat((a.^(2.*exponent./(nVar-1)))', 1, N) .* (z-2).^2 )./repmat(a.*a.*nVar, 1, N);
    feasx = x;
    

function [obj, feasx] = CIGTABnonrotated(x, nObj, penaltyfactor)
    % Cigar-Tablet function, unrotated
    % note that this is an unconstrained benchmark problem
    % according to original MO-CMA-ES paper, this problem has 10 variables
    
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem CIGTABnonrotated only works for 2 objectives');
    end
    a = 1000; % parameter of the function
    
    obj = zeros(2, N);
    obj(1,:) = 1/(a.^2 * nVar) .* (x(1,:).^2 + sum(a.*(x(2:nVar-1,:).^2)) + a.^2.*x(nVar,:).^2);
    obj(2,:) = 1/(a.^2 * nVar) .* ((x(1,:)-2).^2 + sum(a.*((x(2:nVar-1,:)-2).^2)) + a.^2.*(x(nVar,:)-2).^2);
    feasx = x;


function [obj, feasx] = CIGTAB(x, nObj, penaltyfactor)
    % Cigar-Tablet function, rotated
    % note that this is an unconstrained benchmark problem
    % according to original MO-CMA-ES paper, this problem has 10 variables
    
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem CIGTAB only works for 2 objectives');
    end
    if nargin==2
        penaltyfactor=1e-6;
    end
    % rotate coordinate system:
    B = computeRotation(nVar, 1);
    y = zeros(nVar, N);
    
    for i=1:N
        y(:,i) = B*x(:,i);
    end
    
    [obj, feasx] = CIGTABnonrotated(x, nObj, penaltyfactor);
    
    
function [obj, feasx] = CIGTAB2(x, nObj, penaltyfactor)
    % Cigar-tablet function, coordinate system rotated independently for
    % each objective function
    % note that this is an unconstrained benchmark problem
    % according to original MO-CMA-ES paper, this problem has 10 variables
    
    [nVar N] = size(x);
    
    if (nObj ~= 2)
        error('problem CIGTAB2 only works for 2 objectives');
    end
    
    obj = zeros(2, N);
    % rotate coordinate systems:
    B = computeRotation(nVar, 1);
    y = zeros(nVar, N);
    for i=1:N
        y(:,i) = B*x(:,i);
    end
    B2 = computeRotation(nVar, 2);
    z = zeros(nVar, N);
    for i=1:N
        z(:,i) = B2*x(:,i);
    end
    
    a = 1000; % parameter of the function
    
    obj = zeros(2, N);
    obj(1,:) = 1/(a.^2 * nVar) .* (x(1,:).^2 + sum(a.*(x(2:nVar-1,:).^2)) + a.^2.*x(nVar,:).^2);
    obj(2,:) = 1/(a.^2 * nVar) .* ((x(1,:)-2).^2 + sum(a.*((x(2:nVar-1,:)-2).^2)) + a.^2.*(x(nVar,:)-2).^2);
    feasx = x;
    
    
function [obj, feasx] = FON(x, nObj, penaltyfactor)
    % returns objective vectors 'obj' as well as decision vectors projected
    % onto the boundaries ('feasx')
    % for solutions outside the domain boundaries, a penalization term
    % is added.
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    if (nObj ~= 2)
        error('problem FON only works for 2 objectives');
    elseif (nVar ~= 3)
        error('problem FON takes exactly 3 variables');
    end
    bounds = [-4.*ones(nVar,1) 4.*ones(nVar,1)];
    % compute unpenalized objective functions first:
    feasx = feasible(x, bounds);
    obj(1,:) = 1 - exp(-1 .* sum((feasx - 1/sqrt(3)).^2));
    obj(2,:) = 1 - exp(-1 .* sum((feasx + 1/sqrt(3)).^2));
    
    % add penalization term:
    for i=1:nObj
        obj(i,:) = obj(i,:) + penaltyterm(x, feasx, penaltyfactor);
    end
    
function [obj, feasx] = ZDT1(x, nObj, penaltyfactor)
    % returns objective vectors 'obj' as well as decision vectors projected
    % onto the boundaries ('feasx')
    % for solutions outside the domain boundaries, a penalization term
    % is added.
    
    if nargin < 3
        penaltyfactor = 1e-6;
    end
    [nVar N] = size(x);
    if (nObj ~= 2)
        error('problem ZDT1 only works for 2 objectives');
    end
    bounds = [zeros(nVar,1) ones(nVar,1)];
    % compute unpenalized objective functions first:
    feasx = feasible(x, bounds);
    obj(1,:) = feasx(1);
    g = 1 + 9.*(sum(feasx(2:end))/(nVar-1));
    obj(2,:) = g .* (1 - sqrt(feasx(1)./g));
    
    % add penalization term:
    for i=1:nObj
        obj(i,:) = obj(i,:) + penaltyterm(x, feasx, penaltyfactor);
    end
    
function F = MYPB(x, nObj)

m=size(x,2);
x=x';
if nObj==2
    % 2 objectifs:
    F=zeros(2,m);
%     F(2,:)=sum(-10*(exp(-0.2*sqrt(x(:, 1:end-1).^2 + x(:, 2:end).^2))), 2);
    F(1,:)= sum(abs(x).^0.8 + 5*sin(x.^3), 2);
    F(2,:)= ((10.*size(x,2)+sum( x.^2 -10.*cos(x.*2.*pi), 2))/5);% rastrigin
elseif nObj==3
    % 3 objectifs:
    
    F=zeros(3,m);
    F(1,:)= (sum(abs(x).^0.8 + 5*sin(x.^3), 2));
    F(3,:)=(sum(-10*(exp(-0.2*sqrt(x(:, 1:end-1).^2 + x(:, 2:end).^2))), 2));
    F(2,:)= ((10.*size(x,2)+sum( x.^2 -10.*cos(x.*2.*pi), 2))/5);% rastrigin
else
     error('problem MYPB only works for 2 or 3 objectives');
end  
    
function [F, feasx] = MYPBmono(x, nObj, penaltyfactor)

feasx=x;
m=size(x,2);
x=x';
if nObj==2
    % 2 objectifs:
    F=zeros(2,m);
    F(1,:)=sum(-10*(exp(-0.2*sqrt(x(:, 1:end-1).^2 + x(:, 2:end).^2))), 2);
    F(2,:)=sum(-10*(exp(-0.2*sqrt(x(:, 1:end-1).^2 + x(:, 2:end).^2))), 2);
elseif nObj==3
    % 3 objectifs:
    
    F=zeros(3,m);
    F(1,:)= (sum(abs(x).^0.8 + 5*sin(x.^3), 2));
    F(2,:)=(sum(-10*(exp(-0.2*sqrt(x(:, 1:end-1).^2 + x(:, 2:end).^2))), 2));
    F(3,:)= ((10.*size(x,2)+sum( x.^2 -10.*cos(x.*2.*pi), 2))/5);% rastrigin
else
     error('problem MYPB only works for 2 or 3 objectives');
end  
% ########################################
%  the following is copied from BBOB/COCO
% ########################################    
    
function B = computeRotation(DIM, cseed)
    % computes an orthogonal basis
	B = reshape(gauss(DIM*DIM, cseed), DIM, DIM);
    for i = 1:DIM
        for j = 1:i-1
            B(:,i) = B(:,i) - B(:,i)'*B(:,j) * B(:,j);
        end
        B(:,i) = B(:,i) / sqrt(sum(B(:,i).^2));
    end

%---------- pseudo random number generator ------------
function g = gauss(N, mseed)
    % gauss(N, seed)
    % samples N standard normally distributed numbers
    % being the same for a given seed
	r = unif(2*N, mseed); % in principle we need only half
	g = sqrt(-2*log(r(1:N))) .* cos(2*pi*r(N+1:2*N));
	if any(g == 0)
        g(g == 0) = 1e-99;
    end
    
    
function r = unif(N, inseed)
% unif(N, seed)
%    generates N uniform numbers with starting seed

	% initialization
	inseed = abs(inseed);
	if inseed < 1
        inseed = 1;
	end
	aktseed = inseed;
	for i = 39:-1:0
        tmp = floor(aktseed/127773);
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
        if aktseed < 0
            aktseed = aktseed + 2147483647;
        end
        if i < 32
            rgrand(i+1) = aktseed;
        end
    end
	aktrand = rgrand(1);

	% sample numbers
	r = zeros(1,N); % makes the function ten times faster(!)
	for i = 1:N
        tmp = floor(aktseed/127773);
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
        if aktseed < 0
            aktseed = aktseed + 2147483647;
        end
        tmp = floor(aktrand / 67108865);
        aktrand = rgrand(tmp+1);
        rgrand(tmp+1) = aktseed;
        r(i) = aktrand/2.147483647e9;
    end
	if any(r == 0)
        warning('zero sampled(?), set to 1e-99');
        r(r == 0) = 1e-99;
    end



