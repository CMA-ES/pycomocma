if size(bounds, 2) ~= 2 
    bounds = bounds';
end
lb = bounds(:, 1);
rb = bounds(:, 2);
for i =1:size(lb, 1)
if lb(i,1) == -inf
    lb(i,1) = -py.numpy.inf;
else
    lb(i,1) = py.float(lb(i,1));
end
if rb(i,1) == inf
    rb(i,1) = py.numpy.inf;
else
    rb(i,1) = py.float(rb(i,1));
end
end
newbounds = py.list({py.list(lb'), py.list(rb')});