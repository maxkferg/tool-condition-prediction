
clear all
format longEng;
FixedNumCom=3;

wind_speed = textread('wsn.txt','','delimiter',',','headerlines',1);
wind_direction = textread('wdn.txt','','delimiter',',','headerlines',1);
load = textread('Mx11.txt','','delimiter',',','headerlines',1);


%synchronizing wind speed and load
[logic_a,location_b] = ismember(wind_speed(:,1),load(:,1));

A=[];
B=[];
C=[];


%Synchronizing wind speed and load
for i=1:length(logic_a)
    if logic_a(i)==1
        A=[A;wind_speed(i,2)]; %mean wind speed
        B=[B;wind_speed(i,3)]; %wind speed standard deviation
        C=[C;load(location_b(i),5)]; %mean wind turbine load
        %wind speed and load are already synchronized
    end
end

%find error failutre region
[index,value]=find(B~=0);

A=A(index);
B=B(index);
C=C(index);

Wind=[A,B];
Load=C;



x=Wind;
y=Load;

    
K=sort(y);
lb=K(round(length(Load)*0.05));
ub=K(round(length(Load)*0.95));
RangeMax=ub-lb;
    
  

%select only good data
[index,value]=find(lb<=y & y<=ub);



x=Wind(index,:);
y=y(index);


    
    
    
     figure(1)
    hold
    for i=1:length(x)
        plot3(x(i,1),x(i,2),y(i),'o','MarkerFaceColor','k','MarkerEdgeColor','none','MarkerSize',3)
    end
   
    grid on 
    
     xlabel('$x_1$','Interpreter','Latex')
     ylabel('$x_2$','Interpreter','Latex')
     zlabel('$f(x_1,x_2)$','Interpreter','Latex')
     legend('(x,f(x))')
     set(gcf,'position',[100 100 700 700])
     view([330,30])
    xlim([min(x(:,1)),max(x(:,1))])
    ylim([min(x(:,2)),max(x(:,2))])
    
     
%     figure(1)
%     hold
%     for i=1:length(x)
%         plot(x(i,1),x(i,2),'o','MarkerFaceColor',[1-PP(i) 0 0],'MarkerEdgeColor','none','MarkerSize',2)
%         alpha(0.5)
%     end





     %matlab sorce
     [label, model, llh] = emgm(x',FixedNumCom);
     mu=model.mu;
     p=model.weight';
     s=model.Sigma;

   
        P=p;
        MU(:,:)=mu;
        S(:,:,:)=s;
  
 clear alpha
 n=2;
 syms z z1 z2
 z=[z1;z2]
 
 
 

    PP=0;
    for j=1:FixedNumCom
        PP=PP+P(j)*1/(2*pi)^(n/2)/(abs(det(S(:,:,j))))^(1/2)*exp(-1/2*(z-MU(:,j))'*inv(S(:,:,j))*(z-MU(:,j)));
    end
    
    PPP=matlabFunction(PP)
    PDF = @(z) PPP(z(1),z(2));

        
    
    
    
    
    
%     
%      figure(1);
%      plot(x(:,1),x(:,2),'k.')   
%      xlabel('$x_1$','Interpreter','Latex');ylabel('$x_2$','Interpreter','Latex')
%      set(gcf,'position',[100 100 700 700])
    
    %plot marginal PDF

o1=linspace(min(x(:,1)),max(x(:,1)),100);
o2=linspace(min(x(:,2)),max(x(:,2)),100);

dx1=o1(2)-o1(1);
dx2=o2(2)-o2(1);



for m=1:length(o1)
    for n=1:length(o2)
            CP(m,n)=PDF([o1(m);o2(n)]);
    end
end

colormap('jet')
%colormap(flipud(colormap))



figure(2)
hold
plot(x(:,1),x(:,2),'k.')   

%Marginal pdf over U P(a, std)
[XX,YY]=meshgrid(o1,o2); %for PDF
XX=squeeze(XX);YY=squeeze(YY)

    [C,h]=contour(XX,YY,CP',10);
    % Loop through each contour line
    for i = 1:length(h)
         % Set the line width
         set(h(i),'LineWidth',3)
    end

set(gcf,'position',[100 100 700 700])
xlabel('$x_1$','Interpreter','Latex');ylabel('$x_2$','Interpreter','Latex')
legend('data','f_X(x)','Interpreter','Latex')
grid on


    xlim([min(x(:,1)),max(x(:,1))])
    ylim([min(x(:,2)),max(x(:,2))])






   
 %Gaussian process here

 
 meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
 covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
 likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
 
 
 covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
 hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
 exp(hyp2.lik)
 nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)

 
 
 
 x1=linspace(min(x(:,1)),max(x(:,1)),30);
 x2=linspace(min(x(:,2)),max(x(:,2)),30);
 [XX,YY]=meshgrid(x1,x2); %for PDF
%  for i=1:length(x1)
%      for j=1:length(x2)
%          z(100*(i-1)+j,:)=[XX(i),YY(j)];
%      end
%  end
 
 
 
 for i=1:length(x1)
     for j=1:length(x2)
 
        [m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, [x1(i),x2(j)]);
        M(i,j)=m;
        S(i,j)=s2;
     end
 end
 
 
        
    
 figure(2)
 hold
 plot3(x(:,1),x(:,2),y,'x')
 surface(XX,YY,M')
 
 
 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
  
  
  
  
  
  
  
  
  
  
  
%%
%apply GDA (quadratic)

     
cqs = ClassificationDiscriminant.fit(x,y',...
    'DiscrimType','linear');     

K = cqs.Coeffs(1,2).Const;
L = cqs.Coeffs(1,2).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2; 
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','c','LineWidth',2)

K = cqs.Coeffs(2,3).Const;
L = cqs.Coeffs(2,3).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2; 
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','g','LineWidth',2)

K = cqs.Coeffs(3,4).Const;
L = cqs.Coeffs(3,4).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','b','LineWidth',2)

K = cqs.Coeffs(4,5).Const;
L = cqs.Coeffs(4,5).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','r','LineWidth',2)


xlabel('$x_1$','Interpreter','Latex');ylabel('$x_2$','Interpreter','Latex')
     set(gcf,'position',[100 100 700 700])
  grid on

   


  
  
 %estimate the error rate
 
 [~,random_index]=sort(randn(length(x),1));
 x=x(random_index,:);
 y=y(random_index);
 
 x_train=x(1:round(length(x)/2),:);
 y_train=y(1:round(length(x)/2));
 x_test=x(round(length(x)/2)+1:end,:);
 y_test=y(round(length(x)/2)+1:end,1);
 
 
 
%  %kernernal approach
%  
%  for i=1:length(x_train)
%      X_train(i,:)=[x_train(i,1),x_train(i,2),x_train(i,1)*x_train(i,2),x_train(i,1)^2,x_train(i,2)^2];
%  end
%  
%  for i=1:length(x_test)
%      X_test(i,:)=[x_test(i,1),x_test(i,2),x_test(i,1)*x_test(i,2),x_test(i,1)^2,x_test(i,2)^2];
%  end
%  
%  x_train=X_train;
%  x_test=X_test;
%  
%  
%  %train
%  cqs = ClassificationDiscriminant.fit(x_train,y_train,...
%     'DiscrimType','quadratic');  
%  %predict
%  [y_predict, score, cost] = predict(cqs,x_test)
%  accuracy=length(find(y_test == y_predict))/length(x_test);
%  
%  
%  cqs = ClassificationDiscriminant.fit(x_train,y_train,...
%     'DiscrimType','linear');  
%  %predict
%  [y_predict, score, cost] = predict(cqs,x_test);
%  accuracy=length(find(y_test == y_predict))/length(x_test); 
%   


     
 

%total figure
figure(4)
hold
K = cqs.Coeffs(1,2).Const;
L = cqs.Coeffs(1,2).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','k','LineWidth',2,'LineStyle',:)

K = cqs.Coeffs(2,3).Const;
L = cqs.Coeffs(2,3).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','k','LineWidth',2,'LineStyle',:)


K = cqs.Coeffs(3,4).Const;
L = cqs.Coeffs(3,4).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','k','LineWidth',2,'LineStyle',:)

K = cqs.Coeffs(4,5).Const;
L = cqs.Coeffs(4,5).Linear; 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(x(:,1)),max(x(:,1)),min(x(:,2)),max(x(:,2))]);
set(h2,'Color','k','LineWidth',2,'LineStyle',:)

    [C,h]=contour(XX,YY,CP'*dx1*dx2,10);
    % Loop through each contour line
    for i = 1:length(h)
         % Set the line width
         set(h(i),'LineWidth',3)
    end

set(gcf,'position',[100 100 700 700])
xlabel('$x_1$','Interpreter','Latex');ylabel('$x_2$','Interpreter','Latex')
%legend('data','f_X(x)','Interpreter','Latex')
grid on
  