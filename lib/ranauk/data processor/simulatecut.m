%%%Simulate the cutting process in order to calculate the depth of cut for
%%%each block and to find out if each block is climb milling or
%%%conventional milling!

%% Addition to EData: Depth of Cut, Strategy of Cut, Intelligent Length of Cut

for i = 2:size(EData,1)
    PROGRESS2 = 100*i/size(EData,1) %Progress bar for sanity
    count = 1; %Initialize count (for determining #elements cut)
    count2 = 0; %Initialize count2 (also takes depth into account)
    flag = 0; %Initialize flag, determines if area was cut to new depth
    in = zeros(1,N); %Counts elements within rectangle of tool motion
    in1 = zeros(1,N); %Counts elements also within circular extent of tool (final)
    in2 = zeros(1,N); %Discounts elements within circular extent of tool (initial)
    R = 0; L =0; T =0; B=0; %Initialize counters for cutting strategy and radius of cut
    
     if EData{i,13} == 0
        EData(i,13) = EData(i-1,13);
     end
     if EData{i,14} == 0
        EData(i,14) = EData(i-1,14);
     end
     if isempty(EData{i,13}) == 1
        EData(i,13) = EData(i-1,13);
     end
     if isempty(EData{i,14}) == 1
        EData(i,14) = EData(i-1,14);
     end
     if isempty(EData{i,18}) == 1
        EData(i,18) = EData(i-1,18);
     end
    
    EData(i,15) = num2cell(EData{i,13} - EData{i-1,13});%dX
    EData(i,16) = num2cell(EData{i,14} - EData{i-1,14});%dY
    EData(i,17) = num2cell(sqrt(EData{i,15}^2 + EData{i,16}^2));%Length of Cut
    
    if EData{i,15} ~=0 || EData{i,16}~=0 %If the tool moved
        if sum(strcmp(EData(i,:),'G04')) == 0  && sum(strcmp(EData(i,:),'G00')) == 0 %No dwells/rapids
        Xprev=EData{i-1,13}; %Previous X coordinate of tool
        Yprev=EData{i-1,14}; %Previous Y coordinate of tool
        Xnew=EData{i,13}; %Current X coordinate of tool at finish of block
        Ynew=EData{i,14}; %Current X coordinate of tool at finish of block
        Zvalue=EData{i,18}; %Current Z coordinate of tool *not completely fixed
        
            if sum(strcmp(EData(i,:),'G02')) == 0 && sum(strcmp(EData(i,:),'G03')) == 0 %Linear cuts
        getcorners(); %Function to plot bounds of tool path
        removematerial(); %Function to determine elements within tool path being cut and cutting strategy
  
            elseif sum(strcmp(EData(i,:),'G02')) == 1 || sum(strcmp(EData(i,:),'G03')) == 1 
                 if EData{i,5}(1) == 'R'
                R = str2num(EData{i,5}(2:size(EData{i,5},2))); 
                getcircle();
                 end
        
%         if isnan(mode(D(find(D>0))')) == 1
%         Depth(i) = mode(D(find(D>0))');
%         else
%         Depth(i)= mode(D(find(D>0))');
%         end

            end
         
        if T>B % See bottom portion of remove material
            EData{i,19} = 'Climb';
        elseif B>T
            EData{i,19} = 'Conventional';
        else
            EData{i,19} = 'Both';
        end
%     else
%         Depth(i) =  0;
    end
    
        if flag == 1 %If a cut occured, then area of cut is proportional to #elements within cut (count 2)
        IAoC(i) = count2*(ms^2); % Area of Cut
        IdX(i) = range(Cx(cut2));  % Intelligent dX
        IdY(i) = range(Cy(cut2)); % Intelligent dY
        end
            

%Assign calculated values into EData cell, should this be done directly in above if?        
     EData{i,20} = IdX(i);
     EData{i,21} = IdY(i);
     EData{i,22} = Depth(i);
     EData{i,23} = IAoC(i);
     EData{i,24} = EData{i,23} * EData{i,22}; %Area * Depth = Volume of Cut
     
    clear cut
    clear cut2
%     else
%         Depth(i)=0;
    end
% Show cuts occuring on plot
%         F=reshape(currentZ,breadthelements,lengthelements);
%         figure(1)
%         contourf(F',4)
        
clc
end
