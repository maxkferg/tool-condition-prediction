function toolWear = ToolWear(toolnum,partnum,progress)
% Return the human labeled tool wear for tool i, part j
% Uses the progress parameter to interpolate the tool wear between two photos
% Parameters:
%   @toolnum - The tool that was used to cut this part
%   @partnum - The part number 
%   @progress - The amount of progress made on this part on a 0-1 scale
% Wear is returned on a scale of 0->1
    wear = containers.Map(0,[0,0]);
    wear(1) = [0,40,80]; % Done
    wear(2) = [0,60,70,80,90,95,96,97,98,99,100]; % Done
    wear(3) = [0,20,40,45,50,55,65,70,75,80,86,90,94]; % Done
    wear(4) = [0,8,16,24,35,60,65,70,82]; % Done
    wear(5) = [0,50,80]; % Done
    wear(6) = [0,40,60,80]; % Done
    wear(7) = [0,50,60,70,75,85]; % Done
    wear(8) = [0,7,15,20,30,33,34,35,36,38,40,43,46,49,52,55]; % Done
    wear(9) = [0,100];
    wear(10) = [0,20,25,30,35,37,39,43,45,47,50,55]; % Done
    wear(11) = [0,10,20,25,30,35,50,60,65,80,85,95]; %Done
    wear(12) = [0,10,20,30,35,60,70,100]; % Done
    wear(13) = [0,40]; % Done
    wear(14) = [0,10,50]; % Done
    wear(15) = [0,30,90];
   % wear(16) = [0,10,20,30,40,50,60,70,80,90,100];
    wear(17) = [0,10,100];
    wear(18) = [0,20,50,65];
    wear(19) = [0,10,20,40,55,60,65,90];
    wear(20) = [0,8,20,50,60,65]; % Strange behavior at 20
    wear(21) = [0,25,50,80,90,95]; 
    wear(22) = [0,20,50,70,75,80];
    wear(23) = [0,12,20,50,55,60];
    
    wear(25) = [0,10,70]; % Boba test
    wear(26) = [0,5,10,20,100]; % Boba test
    
    %wear(27) = [90,100]; % Boba test 2. Broken from start :(
    wear(28) = [0,10,35,50,75]; % Boba test 2
    wear(29) = [0,10,30,60,70,85]; % Boba test 2
     
    
    partWear = wear(toolnum);
    currWear = partWear(partnum);
    nextWear = partWear(partnum+1);
    
    % Linear interpolation on progress
    toolWear = (1-progress)*currWear + progress*nextWear;
    
    % Sanity check
    if toolWear<currWear || toolWear>nextWear
        error('Wear must increase montonically');
    end
    % Convert to 0-1 scale
    toolWear = toolWear/100;
end

