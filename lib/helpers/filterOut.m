function objects = filterOut(objects,property,value)
% Filter an array of objects, removing those where object.property==value
% @param{Array} objects. An array of objects
% @param{String} property. A property name
% @param{Anything} value. A property value
    for i=length(objects):-1:1
       if (objects(i).(property)==value)
           objects(i) = [];
       end
    end
end


