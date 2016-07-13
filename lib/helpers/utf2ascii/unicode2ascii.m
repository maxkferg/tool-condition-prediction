function varargout = unicode2ascii(sourceFile, varargin)
%UNICODE2ASCII Converts unicode endcoded files to ASCII encoded files
%  UNICODE2ASCII(FILE)
%  Converts the file to ASCII (overwrites the old file!)
%  UNICODE2ASCII(SOURCEFILE, DESTINATIONFILE)
%  Converts the contents of SOURCEFILE to ASCII and writes it to
%  DESTINATIONFILE
%  ASCIISTRING = UNICODE2ASCII('string', UTFSTRING)
%  Converts the UTFSTRING to ASCII and returns the string.

% check number of arguments and open file handles
strout = false;

if(nargin == 1)
    fin = fopen(sourceFile,'r');
elseif(nargin == 2)
    if(strcmpi(sourceFile, 'string'))
        file = varargin{1};
        strout = true;
    else
        fin = fopen(sourceFile,'r');
    end
else
    error('too many input arguments!');
end

%does the file exist?
if (~strout && fin == -1)
    error(['File ' sourceFile ' not found!'])
    return;
end

% read the file and delete unicode characters
if(~strout)
    unicode = isunicode(sourceFile);
    file = fread(fin);
else
    unicode = isunicode('string', file);
end

% delete header
switch(unicode)
    case 1
        file(1:3) = [];
    case 2
        file(1:2) = [];
    case 3
        file(1:2) = [];
    case 4
        file(1:4) = [];
    case 5
        file(1:4) = [];
end

% deletes all 0 bytes
file(file == 0) = [];

if(strout)
    varargout{1} = file;
    return;
end

% write the ascii file
if(nargin == 2)
    fin2 = fopen(varargin{1}, 'w+');
    fwrite(fin2, file, 'uchar');
    fclose(fin2);
else
    fclose(fin);
    delete(sourceFile);
    fin = fopen(sourceFile,'w+');
    fwrite(fin, file, 'uchar');
end
if(~strout)
    fclose(fin);
end