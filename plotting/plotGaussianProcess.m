close all;

meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

n = 20;
x = gpml_randn(0.3, n, 1);
K = feval(covfunc{:}, hyp.cov, x);
mu = feval(meanfunc{:}, hyp.mean, x);
y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);
 
% Plot points as '+'
figure(1)
set(gca, 'FontSize', 24)
disp(' '); disp('plot(x, y, ''+'')')
plot(x, y, '+', 'MarkerSize', 12)
axis([-1.9 1.9 -0.9 3.9])
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f1.eps; end

nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)

z = linspace(-1.9, 1.9, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

figure(2); hold on;
set(gca, 'FontSize', 20)

% Write a constant band of color
m = zeros(length(z),1);
s2 = ones(length(z),1);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8);

% Write 5 different mean functions
rng(43)
nfuncs = 5;
funcs = zeros(length(z),nfuncs);
for i = 1:nfuncs
    n = 20;
    x = linspace(0,2,n)'-1;
    y = 6*rand(n,1)-3;
    hyper.cov = [-0.4774, -0.1092];
    hyper.lik = -1.04;
    [m s2] = gp(hyper, @infExact, [], covfunc, likfunc, x, y, z);
    plot(z,m,'linewidth',2);
    funcs(:,i)=m-mean(m);
end
ylim([-2.4,2.4])
xlim([0,1])
xlabel('Input, x')
ylabel('Output, y')
title('Randomly Sampled Functions')
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   END %%%%%%%%%%%%%%%%%%%%%%%%%



x = x([1,12]);
y = y([1,12]);

% Write random functions to the function space
figure;
disp('hold on; plot(z, m); plot(x, y, ''+'')')
hold on; plot(z, m, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
axis([-1.9 1.9 -0.9 3.9])
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f2.eps; end

disp(' ')
disp('covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);')
covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
disp('hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y)')
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
%hyp2.lik = 0;
%hyp2.cov = [0.4774, -0.1092];
%hyp2.lik = 0;

exp(hyp2.lik)
disp('nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)')
nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)
disp('[m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, z);')
[m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, z);
s2 = s2-0.02;

disp(' ')
figure(3)
set(gca, 'FontSize', 24)
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8)');
fill([z; flipdim(z,1)], f, [7 7 7]/8)
disp('hold on; plot(z, m); plot(x, y, ''+'')');
hold on; 
h1 = plot(z, m, 'LineWidth', 2); 
plot(x, y, '+k', 'MarkerSize', 16)
grid on
xlabel('input, x')
ylabel('output, y')
axis([-1.9 1.9 -0.9 3.9])

% Write on the random functions
% Functions must be scaled centered on the mean, and have the same sd
for i = 1:nfuncs
    n = 20;
    x = linspace(0,2,n)'-1;
    y = (funcs(:,i)).*s2+m;
    plot(z,y,'--','linewidth',1);
end

xlabel('Input, x')
ylabel('Output, y')
title('GP Distribution With Two Data Points')
legend(h1,{'Mean'})
set(gca, 'FontSize', 20)
