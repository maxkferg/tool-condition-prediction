function [E_predict, S_predict ] = predictenergy(epm, data)

    
    Feature_set{1} = [1 2 3   4 5 6 7   8 9 10 ];
    Feature_set{2} = [1 2 3   4 5 6 7   8 9 10 ];
    Feature_set{3} = [1 2 3   4 5 6 7   8 9 10 ];
    Feature_set{4} = [1 2 3   4 5 6 7   8 9 10 ];
    Feature_set{4} = [1 2 3   4 5 6 7   8 9 10 ];
    Feature_set{5} = [1 2 3   4 5 6 7   8 9 10 ];
    Feature_set{6} = [1 2 3   4 5 6 7   8 9 10 ];

    %the trained model is saved as following 
    X = epm.X;
    Y = epm.Y;
    f = epm.F(1,:);%prediction object
    index_job =epm.F(2,:);%data index
    
    %load table
    %data=load(testFile);
    D3 = data;
    if isstruct(D3)
        D2 = struct2cell(D3);
        D = D2{1};
    else
        D=D3;
    end
    
    %extract fetures
    energy = cell2mat(D(:,9)); %answer
    duration = cell2mat(D(:,10)); %duration
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
    volume_cut = cell2mat(D(:,29)); %Depth of cut

%     ratio_cut = actual_length_cut./length_cut_XY;
%     for i=1:length(ratio_cut)
%         if ratio_cut(i)>0;
%             ratio_cut(i)=ratio_cut(i);
%         else
%             ratio_cut(i)=0;
%         end
%     end

    %cut_method=zeros(length(y),1);
for i=1:size(energy,1)

    if strcmp(D{i,24},'Conventional')
        cut_method(i,:) = [1,0,0];   
    elseif strcmp(D{i,24},'Climb')
        cut_method(i,:) = [0,1,0];
    else
        cut_method(i,:)=[0,0,1];
    end 
end

%cut_direction=zeros(length(y),1);
for i=1:size(energy,1)
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
    for i=1:size(feed,1)

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
        elseif strcmp(D{i,30},'Rapid retract')
            label(i,1) = 6;         

        elseif strcmp(D{i,30},'Dwell')
            label(i,1) = 7;  
        else
            label(i,1) = 0;
        end   
    end


    for i=1:size(energy,1)
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
    density=energy./length_cut_XYZ;


    %total data set
    X_test = input;
    ID_test = input(:,12);
    E_test = energy;
    for i=1:size(E_test,1)
        if ID(i) == 5 %dwell
            L_test(i,1)=1;
        else
            if (X_test(i,11) == 0) %length equals 0 then use duration 
                L_test(i,1) = X_test(i,13); %13 for duration
            else
                L_test(i,1)=X_test(i,11);
            end  
        end
    end
    Y_test = E_test./L_test;
    T_test = input(:,13);

    %Learning GP
    
    meanfunc = {@meanSum, {@meanLinear, @meanConst}}; %hyp.mean = zeros(size(X,2)+1,1);
    likfunc = @likGauss; %sn = 500; hyp.lik = log(sn);
    covfunc = @covSEard; 

    if size(E_test)==0
        S_predict=0;
        E_predict=0; 
    else
        for i=1:size(E_test,1)
                        
            %select the right prediction function
            
            process_id = ID_test(i);
            
            if (process_id>0 && process_id <6)
                hyp = f{process_id};
                X_training = X(index_job{process_id}, Feature_set{process_id});
                Y_training = Y(index_job{process_id});
                
                [Y_predict(i,1), S_predict(i,1) ] = gp(hyp, @infExact, [], covfunc, likfunc, X_training, Y_training, X_test(i,Feature_set{process_id}));
                E_predict(i,1) = Y_predict(i,1).*L_test(i);
            else
                Y_predict(i,1)=0;
                S_predict(i,1)=0;
                E_predict(i,1)=0;
            end
        end
    end

end

