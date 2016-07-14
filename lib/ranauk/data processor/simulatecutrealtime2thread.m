%% Simulate the cutting process real-time
clear all
close all
clc

counter=1;
modal=num2cell(0);

%% Initialize GP Machine Learning Module & Training

startup();
clear OCT
disp('Training...')
EP1= epm2([], 'Training1.mat', 1);
trained = 1;
clc

%% Initializations for simulation parameters

prompt={'Enter workpiece length :','Enter workpiece breadth :','Enter workpiece height :','Enter tool diameter :','Enter filename :','Enter machine tool (Left=0,Right=1) :'};
dlg_title = 'Input parameters';
num_lines = 1;
def = {'63.5','63.5','20','9.525','test1','1'};
answer = inputdlg(prompt,dlg_title,num_lines,def);

if isempty(answer{1})
    length = 63.5;
else
    length=ceil(str2num(answer{1})*10)/10; %Least count issue. Temporary.
end

if isempty(answer{2})
    breadth = 63.5;
else
    breadth=ceil(str2num(answer{2})*10)/10;
end

if isempty(answer{3})
    height = 20;
else
    height=ceil(str2num(answer{3})*10)/10;
end

if isempty(answer{4})
    tooldia = 9.525;
else
    tooldia=str2num(answer{4});
end

if isempty(answer{5})
    filename='test1';
else
    filename=answer{5};
end

if isempty(answer{6})
    machine = 1;
else
    machine=str2num(answer{6}); %define Mori_Seiki_left as 0 and Mori_Seiki_right as 1
end

r = tooldia/2;

prompt2={'Enter X origin with respect to left bottom corner [Length/2]:','Enter Y origin with respect to left bottom corner [Breadth/2] :'};
dlg_title2 = 'Setting origin for machine';
num_lines2 = 1;
def2 = {num2str(length/2),num2str(breadth/2)};
answer2 = inputdlg(prompt2,dlg_title2,num_lines2,def2);

if isempty(answer2{1})
    Xorigin = length/2;
else
    Xorigin = str2num(answer2{1});
end

if isempty(answer2{2})
    Yorigin = breadth/2;
else
    Yorigin = str2num(answer2{2});
end

Xshift = Xorigin - length/2;
Yshift = Yorigin - breadth/2;

%% mesh geometry with mesh size 'ms', determines density of element grid
ms=0.1;
tolerance=0.1;

%Calculate number of elements in each direction
lengthelements=length/ms;
breadthelements=breadth/ms;
N = lengthelements*breadthelements; %Total number of elements

%% Preallocation of variables

row = zeros(1,N);
column = zeros(1,N);
LBx = zeros(1,N); %Left bottom x coordinate of element square
LBy = zeros(1,N); %Left bottom y coordinate of element square
RBx = zeros(1,N); %Right botttom x
RBy = zeros(1,N); %Right bottom y
LTx = zeros(1,N); %Left top x
LTy = zeros(1,N); %Left top y
RTx = zeros(1,N); %Right top x
RTy = zeros(1,N); %Right top y
Cx = zeros(1,N); %X coordinate of center of element
Cy = zeros(1,N); %Y coordinate of center of element
currentZ = zeros(1,N); %currentZ value used in determining depth of cut
D = zeros(1,N); %'D' variable used in determining if elements were cut

%% Set origin of the part - default is center of the block



%% Get co-ordinates for each corner of each element L/R = Left/Right, T/B = Top/Bottom, C = center

disp('Creating mesh. . . ');

for element=1:N
    row(element)=1+floor((element-0.5)/lengthelements);
    column(element)=rem((element-1),lengthelements)+1;
    LBx(element)=-(length/2)+(column(element)-1)*ms;
    LBy(element)=-(breadth/2)+(row(element)-1)*ms;
    RBx(element)=LBx(element)+ms;
    RBy(element)=LBy(element);
    LTx(element)=LBx(element);
    LTy(element)=LBy(element)+ms;
    RTx(element)=RBx(element);
    RTy(element)=LTy(element);
    Cx(element) = .5*(LTx(element)+RBx(element));
    Cy(element) = .5*(LTy(element)+RBy(element));
end

clc

%% Simulation loop
while 1
    
    charc=num2str(counter);
    
    if exist(['test/Block_' charc '.txt'],'file')==2 %Change .txt to .xlsx if reading from Excel!
        ANS=importdata(['test/Block_' charc '.txt']);
        if strcmp(ANS,'Done') == 0
            
            [timestamp, blk, dur, E, feedrate, spindlespeed, SL, XL, YL, ZL, Xnew, Ynew, Zvalue, dX, dY, dZ] = textread(['test/Block_' charc '.txt'],'%f %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f', 1);
            %[alldata,blockdata]=xlsread(['Block_' charc '.xlsx']); %If reading
            %from Excel!
            
            blk=num2str(cell2mat(blk));
            %         blk=num2str(cell2mat(blockdata)) %Uncomment if reading from .xlsx
            %         dur=alldata(4);
            %         spindlespeed=alldata(6);
            %         feedrate=alldata(5);
            %         SL=alldata(7);
            %         XL=alldata(8);
            %         YL=alldata(9);
            %         ZL=alldata(10);
            %         E=alldata(3);
            
            %% Get previous and new coordinates
            
            %         Zvalue=alldata(13); %Uncomment if reading from .xlsx
            %         Ynew=alldata(12);
            %         Xnew=alldata(11);
            %         Xprev=Xnew-alldata(14);
            %         Yprev=Ynew-alldata(15);
            %         Zprev=Zvalue-alldata(16);
            Xnew = Xnew + Xshift;
            Ynew = Ynew + Yshift;
            Xprev=Xnew-dX;
            Yprev=Ynew-dY;
            Zprev=Zvalue-dZ;
            
            %% Define and track modal G-codes
            modalprev=modal;
            if isnan(feedrate)==0
                if isempty(strfind(blk,'G02'))~= 1 || isempty(strfind(blk,'G2'))~= 1
                    modal = num2cell(2);
                elseif isempty(strfind(blk,'G03'))~= 1 || isempty(strfind(blk,'G3'))~= 1
                    modal = num2cell(3);
                elseif isempty(strfind(blk,'G01'))~= 1 || isempty(strfind(blk,'G1'))~= 1 || (feedrate>5 && feedrate<2000)
                    modal = num2cell(1);
                elseif isempty(strfind(blk,'G00'))~= 1 || isempty(strfind(blk,'G0'))~= 1 || feedrate>2000
                    modal = num2cell(0);
                else
                    modal=modalprev; % Smooth the modal tracking if none of the above categories
                end
            else
                modal=modalprev; % Smooth the modal tracking if none of the above categories
            end
            
            %% Cutting simulation
            
            %PROGRESS2 = 100*i/size(EData,1) %Progress bar for sanity - not
            %required in real-time data processing
            count = 1; %Initialize count (for determining #elements cut)
            count2 = 0; %Initialize count2 (also takes depth into account)
            flag = 0; %Initialize flag, determines if area was cut to new depth
            in = zeros(1,N); %Counts elements within rectangle of tool motion
            in1 = zeros(1,N); %Counts elements also within circular extent of tool (final)
            in2 = zeros(1,N); %Discounts elements within circular extent of tool (initial)
            R = 0; L =0; T =0; B=0; %Initialize counters for cutting strategy and radius of cut
            RCheck=2; %Initialize RCheck
            
            %         dX=alldata(14); %Uncomment if reading from .xlsx
            %         dY=alldata(15);
            %         dZ=alldata(16);
            LoC = sqrt(dX^2+dY^2);
            
            if dX~=0 || dY~=0 || isempty(strfind(blk,'G83'))~= 1 || isempty(strfind(blk,'G73'))~= 1 || isempty(strfind(blk,'G02'))~= 1 || isempty(strfind(blk,'G03'))~= 1 || isempty(strfind(blk,'G2'))~= 1 || isempty(strfind(blk,'G3'))~= 1 %If the tool moved or drilled or circled
                if (isempty(strfind(blk,'G04'))== 1 && isempty(strfind(blk,'G4'))== 1  && isempty(strfind(blk,'G00'))== 1 && isempty(strfind(blk,'G0'))== 1) || isempty(strfind(blk,'G02'))== 0 || isempty(strfind(blk,'G03'))== 0 || isempty(strfind(blk,'G2'))== 0 || isempty(strfind(blk,'G3'))== 0%No dwells/rapids
                    if isempty(strfind(blk,'G02'))== 1 && isempty(strfind(blk,'G03'))== 1 && isempty(strfind(blk,'G2'))== 1 && isempty(strfind(blk,'G3'))== 1 && isempty(strfind(blk,'G73'))== 1 && isempty(strfind(blk,'G83'))== 1 %Linear cuts
                        getcornersrealtimethread(); %Function to plot bounds of tool path
                        removematerialrealtimethread(); %Function to determine elements within tool path being cut and cutting strategy
                        if isnan(mode(D(find(D>0))')) == 1
                            Depth = 0;
                        else
                            Depth= mode(D(find(D>0))');
                        end
                        
                    elseif isempty(strfind(blk,'G02')) ~= 1 || isempty(strfind(blk,'G03')) ~= 1 || isempty(strfind(blk,'G2')) ~= 1 || isempty(strfind(blk,'G3')) ~= 1 %Circular Cuts
                        if isempty(strfind(blk,'J')) == 0 || isempty(strfind(blk,'I')) == 0
                            RCheck = 0;
                        elseif isempty(strfind(blk,'R')) == 0
                            text2 = blk(2:end-1);
                            [numericChars,alphabeticIdcs] = regexp(text2,'[A-Z]','split');
                            fields = arrayfun(@(x)text2(x),alphabeticIdcs)';
                            n      = size(fields,1);
                            values = mat2cell(str2double(numericChars(2:end))',ones(1,n));
                            RCheck = 1;
                            R = cell2mat(values(find(fields=='R')));
                        end
                        
                        if RCheck>0
                            getcircleRrealtimethread();
                        elseif RCheck==0;
                            getcircleIJKrealtimethread();
                        end
                        
                        if isnan(mode(D(find(D>0))')) == 1
                            Depth = 0;
                        else
                            Depth= mode(D(find(D>0))');
                        end
                        
                    else
                        Depth = 0;
                        IdX=0;
                        IdY=0;
                        IAoC=0;
                        strat=0;
                    end
                    
                    tol = tolerance*count2;
                    if T-B>tol % See bottom portion of remove material
                        strat = 'Climb';
                    elseif B-T>tol
                        strat = 'Conventional';
                    else
                        strat = 'Both';
                    end
                else
                    Depth =  0;
                    IdX=0;
                    IdY=0;
                    IAoC=0;
                    strat=0;
                end
                
                if flag == 1 %If a cut occured, then area of cut is proportional to #elements within cut (count 2)
                    IAoC = count2*(ms^2); % Area of Cut
                    IdX = range(Cx(cut2));  % Intelligent dX
                    IdY = range(Cy(cut2)); % Intelligent dY
                else
                    IAoC = 0; % Area of Cut
                    IdX = 0;  % Intelligent dX
                    IdY = 0; % Intelligent dY
                end
                
                if isempty(strfind(blk,'G83')) == 0 || isempty(strfind(blk,'G73')) == 0 % Drilling
                    if isempty(strfind(blk,'Q')) == 0
                        Depth = cell2mat(regexp(blk,'(?<=Q)\d+(\.\d+)?','match')); % Peck Depth
                    end
                    if isempty(strfind(blk,'R')) == 0
                        ReturnHeight = cell2mat(regexp(blk,'(?<=R)\d+(\.\d+)?','match')); %Return height for if G99
                        if isstr(ReturnHeight)==1
                            ReturnHeight= str2num(ReturnHeight);
                        end
                    end
                    
                    if isempty(strfind(blk,'G83')) == 0 %G83 canned drilling cycle
                        dZ = Zvalue-ReturnHeight;
                        LoC = 8; %hardcoded for now since Z retract not registered!Should we change EData{i,22} here with the retracted Z value?
                        IdX = 0; %IdX
                        IdY = 0; %IdY
                        IAoC = pi*r^2; %Area
                        TLoC = LoC;%Total Length of cut % Simulate similar to a plunge for depth of cut
                        %else for G73?
                    end
                end
                
                clear cut
                clear cut2
                
                %Else If a spiral is detected (dX=0,dY=0 but G02/G2/G03/G3 & I/J/K, then
                %getspiralIJK, if dX=0,dY=0 but G02/G2/G03/G3 & R, then getspiralR
                
            else
                Depth=0;
                IdX=0;
                IdY=0;
                IAoC=0;
                strat=0;
            end
            
            if isempty(strfind(blk,'G83')) == 1 % TLoC for drilling already defined
                TLoC=sqrt(LoC^2+dZ^2);
            end
            
            if dZ < 0 && isempty(strfind(blk,'G83')) == 1 && isempty(strfind(blk,'G73')) == 1 && RCheck==2 %If and only if it is a plunge
                getZcutrealtimethread();
                strat=0;
            end
            %% Assign calculated values into EData cell
            
            %EData{counter,1} = alldata(1); % Timestamp - uncomment for .xlsx
            EData{counter,1} = timestamp;
            EData(counter,2) = {blk};
            EData(counter,9) = {E};
            EData(counter,10) = {dur};
            EData(counter,11) = {feedrate};
            EData(counter,12) = {spindlespeed};
            EData(counter,13) = {SL};
            EData(counter,14) = {XL};
            EData(counter,15) = {YL};
            EData(counter,16) = {ZL};
            EData{counter,17} = Xnew;
            EData{counter,18} = Ynew;
            EData{counter,19} = dX;
            EData{counter,20} = dY;
            EData{counter,21} = LoC;
            EData{counter,22} = Zvalue;
            EData{counter,23} = dZ;
            EData{counter,24} = strat;
            EData{counter,25} = IdX;
            EData{counter,26} = IdY;
            EData{counter,27} = Depth;
            EData{counter,28} = IAoC;
            EData{counter,29} = abs((EData{counter,22})*(EData{counter,21})); %Area * Depth = Volume of Cut
            EData{counter,31} = TLoC; %Length of cut in X-Y-Z
            EData(counter,32) = modal;
            
            
            if isempty(strfind(blk,'G83')) ~= 1 || isempty(strfind(blk,'G73')) ~= 1 %Volume of cut defined differently for drilling
                EData{counter,29} = abs(EData{counter,28}*EData{counter,23}); %Should change to dZ if get proper retract?
            end
            
            
            if sum(strcmp(blk,'G04')) == 1
                EData{counter,30} = 'Dwell';
            elseif EData{counter,29} > 0 && cell2mat(modal)~=0
                EData{counter,30} = 'Cut with Feed';
            elseif cell2mat(modal)==0 && dX ~=0 ||cell2mat(modal)==0 && dY ~=0 || cell2mat(modal)==0 && dZ ~=0
                EData{counter,30} = 'No Cut - Rapid motion';
            elseif cell2mat(modal)==1 && dX ~=0 ||cell2mat(modal)==1 && dY ~=0 || cell2mat(modal)==2 && dX ~=0 && dY ~=0 || cell2mat(modal)==3 && dX ~=0 && dY ~=0
                EData{counter,30} = 'Air-Cut';
            elseif cell2mat(modal)==1 && dZ>0
                EData{counter,30} = 'Air-cut in Z while retracting';
            elseif cell2mat(modal)==0 && dZ>0
                EData{counter,30} = 'Rapid retract';
            end
            
            if cell2mat(modal)==1 && dZ<0 && EData{counter,29} > 0
                EData{counter,30} = 'Plunge with feed';
            elseif cell2mat(modal)==1 && dZ<0 && EData{counter,29} == 0
                EData{counter,30}= 'Air-Cut in Z while plunging';
            end
            
            clear RCheck
            %% Write data to Excel
            
            %delete([filename '_realtime.mat']);
            
            if exist('EData','var')
                firstrow = num2cell(zeros(1,size(EData,2))); %Free first row for column headers
                EDatap = cat(1,firstrow,EData);
                EDatap{1,1} = 'Timestamp (s)'; %See MTC_Data_Log_Transformer for column definitions
                EDatap{1,2} = 'Code';
                EDatap{1,9} = 'Energy (J)';
                EDatap{1,10} = 'Duration (s)';
                EDatap{1,11} = 'Feed rate (mm/min)';
                EDatap{1,12} = 'Spindle speed (RPM)';
                EDatap{1,13} = 'Spindle Load (%)';
                EDatap{1,14} = 'X Load (%)';
                EDatap{1,15} = 'Y Load (%)';
                EDatap{1,16} = 'Z Load (%)';
                EDatap{1,17} = 'X (mm)';
                EDatap{1,18} = 'Y (mm)';
                EDatap{1,19} = 'dX (code) (mm)';
                EDatap{1,20} = 'dY (code) (mm)';
                EDatap{1,21} = 'Length of Cut XY (code) (mm)';
                EDatap{1,22} = 'Z (mm)';
                EDatap{1,23} = 'dZ (mm)';
                EDatap{1,24} = 'Cutting Strategy'; %Climb/Conventional/Both
                EDatap{1,25} = 'IdX (mm)';
                EDatap{1,26} = 'IdY (mm)';
                EDatap{1,27} = 'Depth of Cut(mm)';
                EDatap{1,28} = 'Area of Cut (mm^2)';
                EDatap{1,29} = 'Volume of Cut (mm^3)';
                EDatap{1,30} = 'Cut Description';
                EDatap{1,31} = 'Length of Cut XYZ (code) (mm)';
                EDatap{1,32} = 'Modal G-Code';
                %xlswrite([filename '_realtime.xlsx'],EDatap) %Write to Excel file, specify filename in first input
            end
            %save([filename '_realtime'],'EData') %Overwrite .mat file for Machine Learning
            
            counter=counter+1;
            delete(['test/Block_' charc '.txt']); %Change between .xlsx and .txt
            
            %% Plot functions
            
            %disp('Plotting...')
            hfig=figure(1);
            %set(hfig,'units','normalized','outerposition',[0.1 0.1 0.9 0.9],'Name','LMAS, UC Berkeley : MTConnect in action') %New fullscreen figure
            
            subplot(2,2,1) %First subplot : Cutting simulation - block-by-block progress
%                     if max(max(currentZ))~=min(min(currentZ)) %Ensure it is not a flat surface
%                         F=reshape(currentZ,breadthelements,lengthelements);
%                         contourf(F',4)
%                         axis equal
%                         axis([0 length/ms 0 breadth/ms])
%                         title('Cutting simulation : Block-by-block progress (2D)')
%                     end
            
            partmap=reshape(currentZ,breadthelements,lengthelements);
            maxdepth=abs(min(min(currentZ)));
            
            if EData{counter-1,29}>0
                cubebit=ones(size(partmap,1),size(partmap,2),maxdepth/ms);
                randommatrix=rand(1,maxdepth/ms);
                Colors=ones(size(partmap,1),size(partmap,2),maxdepth/ms);
                for k=1:size(randommatrix,2)
                    Colors(:,:,k)=randommatrix(k);
                end
                Colors(1,:,:)=0.2; Colors(end,:,:)=0.2; Colors(:,1,:)=0.2; Colors(:,end,:)=0.2; Colors(:,:,1)=0.2; Colors(:,:,end-1)=0.7; Colors(:,:,end)=0.7; %Outer cubes are all of same color for homogenity
                
                for i=1:size(partmap,1)
                    for j=1:size(partmap,2)
                        if round(partmap(i,j)/ms)>=0
                        else
                            cubebit(i,j,(end+round(partmap(i,j)/ms)+1):end)=0;
                        end
                    end
                end
            end
            plot(NaN)
            if exist('cubebit','var')
                x=-length/2:0.1:length/2; y=-breadth/2:0.1:breadth/2; z=height-maxdepth:0.1:height; plotSimulation(x,y,z,cubebit,Colors); %%To be optimized
            end
            vert = [-length/2 -breadth/2 0;length/2 -breadth/2 0;length/2 breadth/2 0;-length/2 breadth/2 0;-length/2 -breadth/2 (height-maxdepth);length/2 -breadth/2 (height-maxdepth);length/2 breadth/2 (height-maxdepth);-length/2 breadth/2 (height-maxdepth)];
            fac = [1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8;1 2 3 4;5 6 7 8];
            patch('Vertices',vert,'Faces',fac,'FaceVertexCData',0.2,'FaceColor','flat')
            view(3)
            axis equal vis3d
            title(['NC Block:  ' blk ', Depth :  ' num2str(Depth) ' mm , Feed :  ' num2str(feedrate) ' mm/min , Speed:  ' num2str(spindlespeed) ' RPM'])

            subplot (2,2,2) %Second subplot : X,Y,Z,S Load bars
            hold off
            plot(NaN)
            H = [XL; YL; ZL; SL];
            Xticks = {'X Load'; 'Y Load'; 'Z Load'; 'S Load'};
            Nmax = numel(H);
            for i=1:Nmax
                h = bar(i, H(i));
                if i == 1, hold on, end
                if H(i) < 10
                    col = 'g';
                elseif H(i) < 35
                    col = 'y';
                else
                    col = 'r';
                end
                set(h, 'FaceColor', col)
            end
            set(gca, 'XTickLabel', '')
            ylim([0 100])
            xlabetxt = Xticks;
            ypos2 = -max(ylim)/50;
            text(1:Nmax,repmat(ypos2,Nmax,1),xlabetxt','horizontalalignment','center','verticalalignment','top','Rotation',0,'FontSize',6)
            title('Load as a % of the maximum rating')
            ylabel('% of maximum load capacity')
            set(gca,'FontSize',6)
            
            subplot (2,2,3) %Third subplot : Energy graph
            energydata=EData(:,9);
            [predictionmean(counter-1) predictionstd(counter-1)]=predictenergy2(EP1,EData(counter-1,:));
            Accuracy = 1 - abs(sum(predictionmean)-sum(cell2mat(energydata)))/sum(cell2mat(energydata));
            hold on
            plot(cell2mat(energydata),'b-')
            plot(predictionmean,'r-')
            title('Energy consumption graph')
            xlabel('Blocks')
            ylabel('Energy consumption [J]')
            hl=legend('Measured','Predicted');
            set(hl,'FontSize',6);
            set(gca,'FontSize',6)
            
            subplot (2,2,4)
            showEnergyDensity(EP1,EData(counter-1,:))
            
            clc
            
            trained=0;
        end
    elseif trained == 0
        EP1= epm('Training1.mat',[filename '_realtime.mat'], 1);
        trained = 1;
        counter = 1;
    end
end