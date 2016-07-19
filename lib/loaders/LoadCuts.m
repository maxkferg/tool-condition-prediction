function toolCuts = LoadCuts(toolNums)
% Load all of the cuts from the data directory
% Each cut is assigned a tool wear score based on the lifetime of the tool
% This function may take up to an hour to run the first time, but it's
% result is saved to the cache directory.
    global tools
    
    if ~isa(tools,'containers.Map')
        tools = containers.Map('KeyType','double','ValueType', 'any');
    end
    
    % Cache tool objects
    for tool=toolNums
        % Cache tools to/from hard drive memory
        cache = sprintf('data/cache/tool_%i.mat',tool);
        if ~exist(cache, 'file')
            data = ToolCondition(tool);
            save(cache,'data');
            tools(tool) = data;
            fprintf('Loaded tool %i from raw data\n',tool);
        elseif ~isKey(tools, tool)
            cached = load(cache);
            tools(tool) = cached.data;
            fprintf('Loaded tool %i from mat cache\n',tool);
        else
            fprintf('Loaded tool %i data from memory\n',tool);    
        end
    end
    
    % Return toolCuts
    toolCuts = [];
    for tool=toolNums
        toolCuts = [toolCuts tools(tool).ToolCuts];
    end  
end