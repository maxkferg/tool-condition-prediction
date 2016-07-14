function [] = showenergyDensity(obj, newTestFile)

    plot_variance = @(x,lower,upper,color) set(fill([x,x(end:-1:1)],[upper,fliplr(lower)],color),'EdgeColor','none');
    meanfunc = {@meanSum, {@meanLinear, @meanConst}}; %hyp.mean = zeros(size(X,2)+1,1);
    likfunc = @likGauss; %sn = 500; hyp.lik = log(sn);
    covfunc = @covSEard; 

    %plotting density for only cutting operations
    hyp2 = obj.F{1,1};
    index_job=obj.F{2,1};
    feature_index = [1 2 3   4 5 6 7   8 9 10 ];
    
    X_training=obj.X(index_job,feature_index);
    Y_training=obj.Y(index_job);

    [XXX,YYY] = extractData(newTestFile);
    feed = XXX(:,1);
    spindle = XXX(:,2);
    depth_cut = XXX(:,3);
    cut_direction = XXX(:,4:7);
    cut_method = XXX(:,8:10);
    density = YYY;
    %cut == 1          input(:,4)        input(:,5)         input(:,3)    input(:,2)     input(:,2)    input(:,12)
    index_11 = find(cut_direction(:,2)==1 & cut_method(:,1)==1 & depth_cut == 1 & spindle >1400 & spindle <1600 );

    feed_range = 0:1:1000;
        %feed            spindle                          depth_cut                     cut_direction                   cut_method
    XX_11 = [feed_range',1500*ones(length(feed_range),1), 1*ones(length(feed_range),1),repmat([0 1 0 0],length(feed_range),1), repmat([1 0 0],length(feed_range),1)];

    %figure(1)
    hold
    plot(feed(index_11),density(index_11),'ob','Linewidth',1.5)
    [m,s]=gp(hyp2, @infExact, [], covfunc, likfunc, X_training, Y_training, XX_11);
    plot(feed_range,m,'--r','linewidth',1.5)
    legend('Measured','Predicted mean')
    plot_variance(feed_range,(m-1.96*sqrt(s))',(m+1.96*sqrt(s))','r')
    alpha(0.2)
    xlabel('Feed rate $x_1$ (mm/min)','Interpreter','Latex')
    ylabel('Energy density (J/mm)','Interpreter','Latex')
    xlim([0,500])
    ylim([0,700])
    text(150,400,'Spindle speed = 1,500 (RPM)','Interpreter','Latex')
    box on
    set(gcf,'position',[100,100,500,300])
end

function [ X, Y ] = extractData(fileName)
    
    D = fileName;

    % extract the field
    energy = cell2mat(D(:,9)); %energy consumption
    duration = cell2mat(D(:,10)); %duration of operation
    feed = cell2mat(D(:,11)); %duration of operation
    spindle = cell2mat(D(:,12)); %spindle speed
    length_cut_X = abs(cell2mat(D(:,19))); %code dx
    length_cut_Y = abs(cell2mat(D(:,20))); %code dy
    length_cut_Z = abs(cell2mat(D(:,23))); %code dy
    length_cut_XY = cell2mat(D(:,21)); %code length_cut
    length_cut_XYZ = cell2mat(D(:,31)); %code length_cut
    actual_dx = cell2mat(D(:,25)); %actual dx 
    actual_dy = cell2mat(D(:,26)); %actual dy
    actual_length_cut = sqrt(actual_dx.^2+actual_dy.^2);
    depth_cut = cell2mat(D(:,27)); %Depth of cut
    area_cut = cell2mat(D(:,28)); %Depth of cut
    volume_cut = cell2mat(D(:,29)); %Depth of cut

    %cut_method=zeros(length(y),1);
    for i=1:length(energy)

        if strcmp(D{i,24},'Conventional')
            cut_method(i,:) = [1,0,0];   
        elseif strcmp(D{i,24},'Climb')
            cut_method(i,:) = [0,1,0];
        else
            cut_method(i,:)=[0,0,1];
        end 
    end

    %cut_direction=zeros(length(y),1);
    for i=1:length(energy)
        if abs(cell2mat(D(i,19)))>0 & abs(cell2mat(D(i,20)))==0
            cut_direction(i,:) = [1,0,0,0];   
        elseif abs(cell2mat(D(i,19)))==0 & abs(cell2mat(D(i,20)))>0
            cut_direction(i,:) = [0,1,0,0]; 
        elseif abs(cell2mat(D(i,19)))>0 & abs(cell2mat(D(i,20)))>0
            cut_direction(i,:)=[0,0,1,0]; 
        else
            cut_direction(i,:)=[0,0,0,1]; 
        end

    end

    %label
    for i=1:length(energy)
        if strcmp(D{i,30},'Cut with Feed')
            label(i,1) = 1;
        elseif strcmp(D{i,30},'Plunge with feed')
            label(i,1) = 2;

        elseif strcmp(D{i,30},'Air-Cut')
            label(i,1) = 3;
        elseif strcmp(D{i,30},'Air-Cut in Z while plunging')
            label(i,1) = 4;
        elseif strcmp(D{i,30},'Air-cut in Z while retracting')
            label(i,1) = 5;

        elseif strcmp(D{i,30},'No Cut - Rapid motion')
            label(i,1) = 6;

        elseif strcmp(D{i,30},'Dwell')
            label(i,1) = 7;  

        else
            label(i,1) = 0;
        end   
    end

        for i=1:length(energy)
            % cut with feed
            if ( label(i)==1 ) %cut with feed (x, y z included)
                ID(i,1) = 1;
            elseif ( label(i)==2)% plunge with feed
                ID(i,1) = 2;
            elseif ( label(i)==3 || label(i)==4 || label(i)==5)% air cut in x-y
                ID(i,1) = 3;
            elseif (label(i)==6) %Rapid motion
                ID(i,1)= 4; 
            elseif (label(i)==7) %dwell
                ID(i,1)= 5; 
            else
                ID(i,1) = 6;
            end
        end

    input= [feed,...,        %1
         spindle,...,        %2
         depth_cut,...,      %3
         cut_direction,...,  %4 5 6 7
         cut_method,...,     %8 9 10
         length_cut_XYZ,..., %11
         ID,...,             %12
         duration];          %13
    output =energy;
    %density=energy./length_cut_XYZ;

    %total data set
    X = input;
    E = energy;
    for i=1:length(E)
        if ID(i) == 5 %dwell
            L(i,1)=1;
        else
            L(i,1)=X(i,11);  
        end
    end
    Y = E./L;

    index_zero = find(Y==0);
    index_zero=union([index_zero], [index_zero+1]);
    index_zero=union([index_zero], [index_zero-1]);

    %clean up data
    clean_up_index = setdiff([1:length(energy)],[index_zero]);
    X = X(clean_up_index,:);
    ID = ID(clean_up_index);
    L = L(clean_up_index);
    E = E(clean_up_index);
    Y = Y(clean_up_index);

    %clean up data
    clean_up_index = find(Y > 0 & Y < 3000);
    X = X(clean_up_index,:);
    ID = ID(clean_up_index);
    L = L(clean_up_index);
    E = E(clean_up_index);
    Y = Y(clean_up_index);

end






