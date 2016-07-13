function [hyp, infFunc, meanFunc, covFunc, likFunc, x, y] = parse(self,filename)
% Parse. Parse a PMML file and return the GPR parameters
%    This function returns the hyperparameters in the same form that
%    is used by the GPML package.
%     
%    @param{String} filename. The filepath to the PMML file
%    @param{Struct} hyp. The hyperparameters in the form used by GPML 
%    @output{String} infFunc. The name of the inference function [Exact] 
%    @output{String} meanFunc. The name of the mean kernel function [MeanZero]
%    @output{String} covFunc. The name of the kernel as defined by the PMML spec
%    @output{String} likFunc. The name of the likilihood function [Gaussian]
%    @output{Matrix} x. An m by n matrix containing x training values
%    @output{Matrix} y. An m by 1 column vector containing y values 

    GPMODEL_TAG = 'gaussianprocessmodel';
    TRAINING_TAG = 'traininginstances';
    KERNEL_TAG = 'ardsquaredexponentialkernel';
    INSTANCE_TAG = 'instancefields';
    INLINE_TABLE = 'inlinetable';
    
    document = parseXML(filename);
    model = getChild(document,GPMODEL_TAG);
    kernalSchema = getChild(model,KERNEL_TAG);
    trainingSchema = getChild(model,TRAINING_TAG);
    instanceFields = getChild(trainingSchema,INSTANCE_TAG);
    inlineTable = getChild(trainingSchema,INLINE_TABLE);
    
    
    % We need to allocate a table for the training values
    % We assign the correct column names and initialize the values to zero
    % Eventually we will be able to support different feature labels
    nrows = getAttribute(trainingSchema,'recordcount','double');
    records = table;
    columns = getChildren(instanceFields,'instancefield');
    for i=1:length(columns)
        records(:,i) = table(zeros(nrows,1));
        records.Properties.VariableNames{i} = getAttribute(columns(i),'field');
    end
    
    % Populate the training values
    rows = getValidChildren(inlineTable);
    for i=1:length(rows)
        columns = getValidChildren(rows(i));
        for j=1:length(columns)
            label = columns(j).Name;
            value = str2double(columns(j).Children.Data);
            records(i,label) = table(value);
        end
    end
    
    % Split the table into x and y values
    x = table2array(records(:,1:end-1));
    y = table2array(records(:,end));
       
    % Populate the hyperparameters
    gamma = getAttribute(kernalSchema,'gamma','double');
    noise = getAttribute(kernalSchema,'noisevariance','double');
    lambdaArray = getChild(getChild(kernalSchema,'lambda'),'array');
    n = getAttribute(lambdaArray,'n','int');
    lambda = parseArray(lambdaArray.Children.Data,n);
    
    % Create the hyperparameter object accroding to the GPML spec
    hyp.mean = [];
    hyp.lik = log(sqrt(noise));
    hyp.cov = [log(lambda) log(sqrt(gamma))];
    
    % Set the functions to their default values
    meanFunc = 'MeanZero';
    infFunc = 'Exact';
    covFunc = 'ARDSquaredExponentialKernel';
    likFunc = 'Gaussian';
    
    % Make sure that the parameters were loaded correctly
    validateParameters(hyp, infFunc, meanFunc, covFunc, likFunc, x, y);
end



function validateParameters(hyp, infFunc, meanFunc, covFunc, likFunc, x, y)
% Perform some simple validation to ensure that the parametes were
% loaded correctly. Throw an error on failure
    nparameters = size(x,2)+1;
    if length(hyp.cov)~=nparameters
        error('There should be 3 kernel parameters, not %i',length(hyp.cov));
    end
    if length(hyp.lik)~=1
        error('There should be 1 noise parameter, not %i',length(hyp.lik));
    end
    if ~strcmp(infFunc,'Exact')
        error('Unsupported inference function %s',infFunc);
    end
    if ~strcmp(meanFunc,'MeanZero')
        error('Unsupported mean function %s',meanFunc);
    end
    if ~strcmp(likFunc,'Gaussian')
        error('Unsupported mean function %s',likFunc);
    end
    if isempty(covFunc)
        error('Kernel function (covariance) is missing');
    end
    if isempty(x)
        error('Training values (x) are missing');
    end
    if isempty(y)
        error('Training values (y) are missing');
    end
    if size(x,2)==size(y,2)
        error('Different number of x and y training pairs');
    end
end



function document = parseXML(filename)
% PARSEXML Convert XML file to a MATLAB structure.
    try
       tree = xmlread(filename);
    catch
       error('Failed to read XML file %s.',filename);
    end

    % Recurse over child nodes. This could run into problems 
    % with very deeply nested trees.
    try
       document = parseChildNodes(tree);
    catch
       error('Unable to parse XML file %s.',filename);
    end
end
     
        
function children = parseChildNodes(theNode)
% Return a multi-element struct containing children on this node.
% Each child will be in the form described by makeStructFromNode
    children = [];
    if theNode.hasChildNodes
        childNodes = theNode.getChildNodes;
        numChildNodes = childNodes.getLength;
        allocCell = cell(1, numChildNodes);

        children = struct(             ...
          'Name', allocCell, 'Attributes', allocCell,    ...
          'Data', allocCell, 'Children', allocCell);

        for count = 1:numChildNodes
            theChild = childNodes.item(count-1);
            children(count) = makeStructFromNode(theChild);
        end
    end
end



function nodeStruct = makeStructFromNode(theNode)
% Return a structure that maps element properties to thier values
% @output{Struct} nodeStruct.Name - The tag name
% @output{Struct} nodeStruct.Attributes - a struct containing name:value pairs
% @output{Struct} nodeStruct.Data - a struct of data
% @output{Array<Struct>} nodeStruct.Children - a cell array of child elements
    nodeStruct = struct(                        ...
       'Name', char(theNode.getNodeName),       ...
       'Attributes', parseAttributes(theNode),  ...
       'Data', '',                              ...
       'Children', parseChildNodes(theNode));

    if any(strcmp(methods(theNode), 'getData'))
       nodeStruct.Data = char(theNode.getData); 
    else
       nodeStruct.Data = '';
    end
end



function attributes = parseAttributes(theNode)
% Return a structure that maps attributes to their values
    attributes = [];
    if theNode.hasAttributes
       theAttributes = theNode.getAttributes;
       numAttributes = theAttributes.getLength;
       allocCell = cell(1, numAttributes);
       attributes = struct('Name', allocCell, 'Value', allocCell);

       for count = 1:numAttributes
          attrib = theAttributes.item(count-1);
          attributes(count).Name = char(attrib.getName);
          attributes(count).Value = char(attrib.getValue);
       end
    end
end


function array=parseArray(strArray,n)
% Parse a space-separated numerical array
% Return a row vector of floats. 
% @param{String} strArray. A string containing only spaces and numeric characters
% @param{Int} n. The expected number of elements in the array 
    next = '';
    position = 1;
    array = zeros(1,n);
    for i=1:length(strArray)
        if (strArray(i)~=' ')
            next = [next strArray(i)];
        else
            array(position) = str2double(next);
            position = position+1;
            next = '';
        end
    end
    % We also need to parse the last value
    array(position) = str2double(next);
            
    % Simple sanity check to make sure everything went smoothly 
    if length(array)~=n;
        error('The number of elements in the array did not match n')
    end
end
    
    
function value=getAttribute(element,attributeName,type)
% Return the value of element.@attribute
%   @param{Struct} element. The DOM element as a struct
%   @param{Struct} attributename. The attribute to return
%   @param{String} type. An optional type to convert. Defaults to string
    for i=1:length(element.Attributes)
        if strcmp(element.Attributes(i).Name, attributeName)
            value = element.Attributes(i).Value;
            if nargin==2
                return
            elseif strcmp(type,'double')
                value = str2double(value);
            elseif strcmp(type,'float')
                value = str2double(value);
            else strcmp(type,'int')
                value = floor(str2double(value));
            end
            return
        end
    end
    error('Could not find attribute %s',attributeName);
end


function child=getChild(parent,name)
% Get a child node from its parent (struct) by tag name
% Throw an error if the node does not exist
    for candidate = parent.Children
       if strcmp(candidate.Name,name)
           child = candidate;
           return;
       end
    end
    error('Could not find element %s',name);
end
           
           

function children=getChildren(parent,tagName)
% Get all children nodes that matches a certain tagName
% Return an empty struct if there are no matching children
    children = parent.Children;
    for i=length(parent.Children):-1:1
        candidate = parent.Children(i);
        if (~strcmp(candidate.Name,tagName))
            children(i) = [];
        end
    end
end

function children=getValidChildren(parent)
% Get all children nodes, excluding text nodes
% Return an empty struct if there are no matching children
    ignore = '#text';
    children = parent.Children;
    for i=length(parent.Children):-1:1
        candidate = parent.Children(i);
        if (strcmp(candidate.Name,ignore))
            children(i) = [];
        end
    end
end

    
    