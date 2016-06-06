addpath matlab;

% compile with built-in function
vl_compilenn('enableImreadJpeg', true, 'enableGpu', true, 'cudaRoot', '/usr/local/cuda',...
             'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', 'local/cudnn');
