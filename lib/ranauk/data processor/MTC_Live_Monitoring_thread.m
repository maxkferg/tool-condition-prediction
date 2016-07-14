% Monitoring Mori Seiki in Student Machine Shop
% UC Berkeley - LMAS

close all
clear all;
clc;

%% Basic initialization

AgentAddress = 'http://machineshop.dyndns.biz:5000/current';
%AgentAddress = 'C:\Users\dhanajay\Desktop\current.xml'; %test static xml file

machine=1; %Choose 1 for right machine and 0 for left machine
Duration = 10; % Machining Duration in Minutes
Store_Range = 100; % Choose Storage Range in Seconds: (initial buffer)

%% Agent Address Validation 

try XML = xmlread(AgentAddress);
catch e
    error('Invalid MTConnect Agent Address!');
end

%% Set machine for XML parsing
if machine==1 %define Mori_Seiki_left as 0 and Mori_Seiki_right as 1
    k=1;
else %extend elseif statements if more machines are added!
    k=0;
end

%% Initialize Variables

tic; %Start counting

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
powersequence=cell(2,1);

%Feed, Speed, Time
Feed = NaN(Store_Range/2,1);
Spindle = NaN(Store_Range/2,1);
Time = NaN(Store_Range/2,1);
blocktimestamp = NaN(Store_Range/2,1);

%Block
Block = cell(Store_Range/2,1);
blocksequence=cell(2,1);
blockcount=1;
CalibrationFactor = 0.046;
%% Read Data
XML = xmlread(AgentAddress);
    
    % Part count
    data = XML.getElementsByTagName('PartCount');
    partcountinitial = data.item(k).getFirstChild.getData;

while toc<60*Duration
    clc
    PROGRESS=(toc/(Duration*60))*100
    XML = xmlread(AgentAddress);
    
    % Execution
    data = XML.getElementsByTagName('Execution');
    Exec = data.item(k).getFirstChild.getData;

    if strcmp(Exec,'ACTIVE')
        
        % Block
        data = XML.getElementsByTagName('Block');
        att = data.item(k).getAttributes;
        blocksequence{2} = char(att.item(2).getTextContent);
        if strcmp(blocksequence{2},blocksequence{1}) && blockcount>1%If block is the same, continue parsing, else finalize current block and initialize for new block
                        
        % Axes Position
%         data = XML.getElementsByTagName('Position');
%         X(1)=[];
%         Y(1)=[];
%         Z(1)=[];
%         X(end+1) = str2num(data.item(0).getFirstChild.getData);
%         Y(end+1) = str2num(data.item(1).getFirstChild.getData);
%         Z(end+1) = str2num(data.item(2).getFirstChild.getData);
        
        % Load
        data = XML.getElementsByTagName('Load');
        LX(1)=[];
        LY(1)=[];
        LZ(1)=[];
        LS(1)=[];
        LX(end+1) = str2num(data.item(4*k+1).getFirstChild.getData);
        LY(end+1) = str2num(data.item(4*k+2).getFirstChild.getData);
        LZ(end+1) = str2num(data.item(4*k+3).getFirstChild.getData);
        LS(end+1) = str2num(data.item(4*k).getFirstChild.getData);
        sload(blockcount)=(sload(blockcount)+LS(end))/2;
        xload(blockcount)=(xload(blockcount)+LX(end))/2;
        yload(blockcount)=(yload(blockcount)+LY(end))/2;
        zload(blockcount)=(zload(blockcount)+LZ(end))/2;
        
        % Real Power
        data = XML.getElementsByTagName('WattageTimeSeries');
        att = data.item(3*k+1).getAttributes;
        powersequence{2} = char(att.item(3).getTextContent);
        if strcmp(powersequence{2},powersequence{1})
        else
            P1(1)=[];
            P2(1)=[];
            P3(1)=[];
            P(1)=[];
            P1(end+1) = sum(str2num(data.item(3*k).getFirstChild.getData));
            P2(end+1) = sum(str2num(data.item(3*k+1).getFirstChild.getData));
            P3(end+1) = sum(str2num(data.item(3*k+2).getFirstChild.getData));
            P(end+1) = (P1(end) + P2(end) + P3(end));
            powersequence{1}=powersequence{2};
            energy(blockcount)=energy(blockcount)+P(end)*0.01*CalibrationFactor;
        end
        
        % Path Feedrate
        data = XML.getElementsByTagName('PathFeedrate');
        Feed(1)=[];
        if k==1
            Feed(end+1) = str2num(data.item(6).getFirstChild.getData); %complicated coz of xml structure mistake by SI config
        else
            Feed(end+1) = str2num(data.item(3*k+1).getFirstChild.getData);
        end
        
        if feed(blockcount)==0
            feed(blockcount)=Feed(end);
        else
            feed(blockcount)=(feed(blockcount)+Feed(end))/2;
        end
        
        % Spindle Speed
        data = XML.getElementsByTagName('SpindleSpeed');
        Spindle(1)=[];
        Spindle(end+1) = str2num(data.item(2*k).getFirstChild.getData);
        if speed(blockcount)==0
            speed(blockcount)=Spindle(end);
        else
            speed(blockcount)=(speed(blockcount)+Spindle(end))/2;
        end
                
        else %If block is different!
            %Finalize current block calculations
            block{blockcount}=Block(end); %skipped in Python
            
            %Get block timestamps
            text=char(att.item(3).getTextContent);
            hour = 10* str2num(text(12)) + str2num(text(13));
            minute = 10* str2num(text(15)) + str2num(text(16));
            second = 10* str2num(text(18)) + str2num(text(19));
            thousandth = 100* str2num(text(21)) + 10* str2num(text(22))+1*str2num(text(23));
            blocktimestamp(1) = [];
            blocktimestamp(end+1) =  3600*hour + 60*minute + second + (1/1000)*thousandth;
            timestamp(blockcount)=blocktimestamp(end);
            
            %Get block durations
            if blockcount > 1
                duration(blockcount)=timestamp(blockcount)-timestamp(blockcount-1);
                
                %Set small feeds at dwells to be zero
                if isempty(strfind(block{blockcount},'G04'))== 0 %skipped in python
                    if feed(blockcount) < 5
                        feed(blockcount) = 0;
                    end
                end
            end
            
            %Extract X, Y, Z positions
            if blockcount>1
                if isempty(strfind(num2str(cell2num(block{blockcount})),'G04'))
                    text = num2str(cell2num(block{blockcount}));
                    xposition=strfind(text,'X');
                    yposition=strfind(text,'Y');
                    zposition=strfind(text,'Z');
                    [numericChars,alphabeticIdcs] = regexp(text,'[A-Z]','split');
                    fields = arrayfun(@(x)text(x),alphabeticIdcs)';
                    n      = size(fields,1);
                    values = mat2cell(str2double(numericChars(2:end))',ones(1,n));
                    if isempty(xposition)==0
                        x(blockcount) = cell2num(values(find(fields=='X')));
                        xset=0;
                    end
                    if isempty(yposition)==0
                        y(blockcount) = cell2num(values(find(fields=='Y')));
                        yset=0;
                    end
                    if isempty(zposition)==0
                        z(blockcount) = cell2num(values(find(fields=='Z')));
                        zset=0;
                    end
                end
            end
            
            %Smooth X, Y, Z positions
            if blockcount>1
                if xset
                    x(blockcount)=x(blockcount-1);
                end
                if yset
                    y(blockcount)=y(blockcount-1);
                end
                if zset
                    z(blockcount)=z(blockcount-1);
                end
            elseif blockcount==1
                x(blockcount)=0;
                y(blockcount)=0;
                z(blockcount)=0;
            end
            
            xset=1;
            yset=1;
            zset=1;
            
            %dX, dY, dZ calculations
            if blockcount > 1
                dx(blockcount)=x(blockcount)-x(blockcount-1);
                dy(blockcount)=y(blockcount)-y(blockcount-1);
                dz(blockcount)=z(blockcount)-z(blockcount-1);
            end

            %Concatenate data
            if blockcount==1
                Datamax=[timestamp(blockcount) block{blockcount} 0 0 0 0 0 0 0 0 x(blockcount) y(blockcount) z(blockcount) 0 0 0];
            else
                Datamax=[timestamp(blockcount) block{blockcount} duration(blockcount) energy(blockcount) feed(blockcount) speed(blockcount) sload(blockcount) xload(blockcount) yload(blockcount) zload(blockcount) x(blockcount) y(blockcount) z(blockcount) dx(blockcount) dy(blockcount) dz(blockcount)];
            end

            xlswrite(['Block_' num2str(blockcount) '.xlsx'],Datamax)
            
            %Update block number
            blockcount=blockcount+1;
            
            %Get new block
            data = XML.getElementsByTagName('Block');
            att = data.item(k).getAttributes;
            Block(1)=[];
            Block(end+1) = data.item(k).getFirstChild.getData;
            blocksequence{1} = blocksequence{2};
            
            %Initialize calculations for next block
            energy(blockcount)=0;
            feed(blockcount)=0;
            speed(blockcount)=0;
            sload(blockcount)=0;
            xload(blockcount)=0;
            yload(blockcount)=0;
            zload(blockcount)=0;
            
        end

    end
    
end
%% Calculate the number of parts manufactured within the duration selected
XML = xmlread(AgentAddress);
    
    % Execution
    data = XML.getElementsByTagName('PartCount');
    partcountfinal = data.item(k).getFirstChild.getData;
    parts=str2num(partcountfinal)-str2num(partcountinitial)