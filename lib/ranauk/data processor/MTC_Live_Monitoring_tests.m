% Monitoring Mori Seiki in Student Machine Shop
% UC Berkeley - LMAS

close all
clear all;
clc;

AgentAddress = 'http://machineshop.dyndns.biz:5000/current';

% Machining Duration in Minutes
Duration = .25;

% Choose Storage Range in Seconds:, should this be same as duration?
Store_Range = 100;

%% Initializations for simulatecut, do before everything else?

length=input('What is the length of the workpiece in X?');
breadth=input('What is the length of the workpiece in Y?');
height=input('What is the height of the workpiece in Z?');
tooldia=input('What is the diameter of the tool?');

r = tooldia/2;

%mesh geometry with mesh size 'ms', determines density of element grid
ms=0.1;

%Calculate number of elements in each direction
lengthelements=length/ms;
breadthelements=breadth/ms;
N = lengthelements*breadthelements; %Total number of elements

% Preallocation of variables

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
IAoC = zeros(size(EData,1),1); %Intelligent area of cut
IdX = zeros(size(EData,1),1); %Inteliigent (only along workpiece) length of cut in x-direction
IdY = zeros(size(EData,1),1); %Intelligent length of cut in y-direction
%Depth = NaN(size(a,1),1); %Should not be needed!!!

%get co-ordinates for each corner of each element assuming origin of
%workpiece in the center! Left/Right, Top/Bottom, C = center
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




%% Initialize Variables

tic;

% %Time
% t = (2-Store_Range:2:0)';

%Position
X = NaN(Store_Range/2,1);
Y = NaN(Store_Range/2,1);
Z = NaN(Store_Range/2,1);
%Load
LX = NaN(Store_Range/2,1);
LY = NaN(Store_Range/2,1);
LZ = NaN(Store_Range/2,1);
LS = NaN(Store_Range/2,1);
%Power
P1 = NaN(Store_Range/2,1);
P2 = NaN(Store_Range/2,1);
P3 = NaN(Store_Range/2,1);
P = NaN(Store_Range/2,1);

%Feed, Speed, Time
Feed = NaN(Store_Range/2,1);
Spindle = NaN(Store_Range/2,1);
Time = NaN(Store_Range/2,1);

%Block
Block = NaN(Store_Range/2,1);

CalibrationFactor = 0.046;

%% Read Data

while toc < 60*Duration
 
    XML = xmlread(AgentAddress);

    X(1)=[];
    Y(1)=[];
    Z(1)=[];

    LX(1)=[];
    LY(1)=[];
    LZ(1)=[];
    LS(1)=[];

    P1(1)=[];
    P2(1)=[];
    P3(1)=[];
    P(1)=[];


    Feed(1)=[];
    Spindle(1)=[];
    
    Block(1)=[];

    Time(1) = [];

    %Time
data = XML.getElementsByTagName('Header');
att = data.item(0).getAttributes;
text = char(att.item(1).getTextContent);
hour = 10* str2num(text(12)) + str2num(text(13));
minute = 10* str2num(text(15)) + str2num(text(16));
second = 10* str2num(text(18)) + str2num(text(19));
Time(end+1) =  3600*hour + 60*minute + second;
    
    % Axes Position
    data = XML.getElementsByTagName('Position');
    X(end+1) = str2num(data.item(0).getFirstChild.getData);
    Y(end+1) = str2num(data.item(1).getFirstChild.getData);
    Z(end+1) = str2num(data.item(2).getFirstChild.getData);
    



    % Load
    data = XML.getElementsByTagName('Load');
    LX(end+1) = str2num(data.item(1).getFirstChild.getData);
    LY(end+1) = str2num(data.item(2).getFirstChild.getData);
    LZ(end+1) = str2num(data.item(3).getFirstChild.getData);
    LS(end+1) = str2num(data.item(0).getFirstChild.getData);


    % Real Power
    data = XML.getElementsByTagName('WattageTimeSeries');
    P1(end+1) = mean(str2num(data.item(0).getFirstChild.getData))*CalibrationFactor;
    P2(end+1) = mean(str2num(data.item(1).getFirstChild.getData))*CalibrationFactor;
    P3(end+1) = mean(str2num(data.item(2).getFirstChild.getData))*CalibrationFactor;
    P(end+1) = (P1(end) + P2(end) + P3(end));
    

    % Path Feedrate
    data = XML.getElementsByTagName('PathFeedrate');
    Feed = str2num(data.item(1).getFirstChild.getData);
    

    % Spindle Speed
    data = XML.getElementsByTagName('SpindleSpeed');
    Spindle = str2num(data.item(0).getFirstChild.getData);

    
    
    % Block
    data = XML.getElementsByTagName('Block');
    Block = str2num(data.item(0).getFirstChild.getData);
    
    
end
%% All together for smoothing
Data = [Time Block P Feed Spindle LS LX LY LZ X Y Z];

%% Smoothing

for c = 3:12                   
    
    for l = 2:size(Data,1)
        
        if isnan(Data(l,c))
        Data(l,c) = Data(l-1,c);
        end
       
    end
end

clear c
clear l

%% Block Processing

%Modify following for loops into if statements that will work in real time

%Pull NC code for each block in cell array, up to 7 code inputs.
%This whole thing must be adapted to new data structure
k=0; 
for line = 2:size(Data,1)
      if strcmp(log(line,2),'block') % Will need new condition
        k=k+1;
         for i =1:7
            BlockCode{k,i} = log{line,i+2};
         end    
      end
end


%find where blocks are ex a = find(Block>0)
for i = 2:size(a,1)
    BlockTime(i-1) = Time(a(i-1));
    Block(i-1) = Block(a(i-1));
    BlockCode(i-1) = BlockCode(a(i-1));
    BlockEnergy(i-1) = sum(Data(a(i-1),3):Data(a(i),3));
    BlockFeed(i-1) = mean(Data(a(i-1):a(i),4));
    BlockSpindle(i-1) = mean(Data(a(i-1):a(i),5));
    BlockLS(i-1) = mean(Data(a(i-1):a(i),6));
    BlockLX(i-1) = mean(Data(a(i-1):a(i),7));
    BlockLY(i-1) = mean(Data(a(i-1):a(i),8));
    BlockLZ(i-1) = mean(Data(a(i-1):a(i),9));
    BlockX(i-1) = Data(a(i-1),10);
    BlockY(i-1) = Data(a(i-1),11);
    BlockZ(i-1) = Data(a(i-1),12);
end

%xyz -> 0
    if isempty(EData{1,18}) == 1
        EData{1,18} = 0;
     end
     if isempty(EData{1,13}) == 1
        EData{1,13} = 0;
     end
     if isempty(EData{1,14}) == 1
        EData{1,14} = 0;
     end
% Combine data into buffer array, call it EData

%% Simulate Cut
%send data to simulate cut line by line with buffer
%if size(EData,1) > 5
%simulatecut();

%% Send to Machine Learning Algorithm 
% GP();
