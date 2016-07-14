X=0:0.1:30;
Y=0:0.01:5;
[XX,YY]=meshgrid(X,Y);
for i=1:501
    for j=1:301
        z(501*(i-1)+j,:)=[XX(i,j),YY(i,j)];
    end
end


  hyp.cov = [0; 0]; hyp.mean = [0; 0; 0]; hyp.lik = log(0.1);
  hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
  [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
  
  
  figure(1)
  hold
  plot3(x(:,1),x(:,2),y,'*r')
  plot3(z(:,1),z(:,2),m,'bo')

  