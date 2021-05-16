clt;
close all;
clear all;
warning off;
%%A CONTINUACION SE SELECCIONA EL NOMBRE DE LA CARPETA DONDE ESTAN CONTENIDAS LAS IMAGENES
st = 'imagenes';%carpeta donde se encuentran las fotos

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fotos_pares = [];
fotos_impares = [];
d = dir([st '/*.png']);
index    = randperm(numel(d), length(d));%%Necesario para funcionalidad de codigo leyendo en desorden los archivos de la carpeta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
for i=1:length(d)
    %full_name = d(index(i)).name %%Descomentar si se quiere probar funcionalidad de codigo leyendo en desorden los archivos de la carpeta
    full_name = d(i).name;
    full_name_cut = strsplit(full_name,'_'); %%array de 3 elementos. El primero es 'face', el segundo es el ID y el tercero es el número de la foto de la persona xxx
    ID = str2double(full_name_cut(2)); %%declaramos especificamente la variable ID y la vamos reescribiendo iteracion a iteracion
    full_name_cut_dot = strsplit(full_name_cut{3},'.'); %%nuevamente volvemos a hacer un split para obtener el numero de la foto de la persona xxx
    char_num_foto = full_name_cut_dot{1};%%declaramos especificamente la variable num_foto y la vamos reescribiendo iteracion a iteracion
    num_foto=str2double(char_num_foto);%%num_foto era un char en la linea previa, por lo que la transformamos a una variable tipo doble
    if(num_foto<=7 && (rem(ID,2)~=0)) %% si es que el numero de la foto de la persona xxx es menor a 7, y ademas el ID es impar
        directorio = [st '/' d(i).name];
        %directorio = [st '/' d(index(i)).name] %%Descomentar si se quiere probar funcionalidad de codigo leyendo en desorden los archivos de la carpeta
        %A continuacion se comienza a trabajar con estructuras, lo cual le entrega más orden al código, y es más elegante
        estruct = struct('addr',directorio, 'label', ID,'numfoto',num_foto);
        fotos_impares = [fotos_impares estruct] ; 
    elseif(num_foto<=7 && (rem(ID,2)==0))%% si es que el numero de la foto de la persona xxx es menor a 7, y ademas el ID es par
        directorio = [st '/' d(i).name];
        %directorio = [st '/' d(index(i)).name];%%Descomentar si se quiere probar funcionalidad de codigo leyendo en desorden los archivos de la carpeta
        estruct = struct('addr',directorio, 'label', ID,'numfoto',num_foto);
        fotos_pares = [fotos_pares estruct] ; 
    end
end

%%%%%%%%%%Condiciones para realizar LBP%%%%%%%%%%%%%%%%%%%%%%
opLBP.vdiv        = 4;           
opLBP.hdiv        = 4;   %Se utiliza una particion de 4x4    
opLBP.samples     = 8;   % number of neighbor samples
opLBP.mappingtype = 'u2'; % Patrones uniformes
opLBP.weight      = 0; 
opLBP.type        = 2; 
D                 = 59; %Al realizar el mapeo se obtendran 59 elementos  
%%%%%%%%%%%%%%%%%%%%%%%DIMENSIONES MATRICES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_samples_Ytrain = 6; %Es el numero de fotos de cada clase. Por ejemplo, la clase k (o individuo k) tendra 6 fotos.Dicho valor me alterara el numero de filas que existiran en cada matriz 
num_samples_Xtrain = 6; 
num_samples_Xtest = 1; 
num_samples_Ytest = 1; 

num_columnas = opLBP.vdiv*opLBP.hdiv*D;%LBP de 59 elementos con 16 particiones en este caso, lo cual entrega columnas de 944 elementos

num_filas_Ytrain = 50*num_samples_Ytrain; %se tendran 50 clases, es decir, 50 individuos. Idem para las 3 lineas de codigo siguiente
num_filas_Xtrain = 50*num_samples_Xtrain;
num_filas_Xtest = 50*num_samples_Xtest; 
num_filas_Ytest = 50*num_samples_Ytest; 
%%%%%%%%%%%%%%%%%%%%%%%INICIALIZACION DE MATRICES IMPARES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ytrain_impar = zeros(num_filas_Ytrain,1); % matriz que contendra los labels correspondientes a Xtrain
Xtrain_impar = zeros(num_filas_Xtrain,num_columnas);%matriz a la que se le hara el entrenamiento con features seleccionados del LBP.
Xtest_impar = zeros(num_filas_Xtest,num_columnas);%matriz que sera usada para el test 
Ytest_impar = zeros(num_filas_Ytest,1); %matriz que contendra los labels correspondientes a Xtest
%%%%%%%%%%%%%%%%%%%%%%%INICIALIZACION DE MATRICES PARES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ytrain_par = zeros(num_filas_Ytrain,1); %idem caso par
Xtrain_par = zeros(num_filas_Xtrain,num_columnas);
Xtest_par = zeros(num_filas_Xtest,num_columnas); 
Ytest_par = zeros(num_filas_Ytest,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=256; 
N=256;%%M y N son variables para el resize de 256x256 pixeles.
contador1=0;
contador2=0;
for k=1:length(fotos_impares)%comienzo a recorrer el total de filas dado por 350 filas (50 clases * 7 num_fotos)
    fotos_impares(k).addr; %obtengo la ubicacion/path de la clase o individuo en la iteracion k
    I = imread(fotos_impares(k).addr);%procedo a 'leer' la imagen contenida en el archivo
    I = imresize(I,[N M]);%Se hace un resize de 256x256 pixeles (notar del dataset que hay imagenes muy pequeñas)
    x_vect = Bfx_lbp(I,[],opLBP); %me entrega una matriz de 1 fila y 944 columnas (944 = 59 elementos*16 particiones)
    if(fotos_impares(k).numfoto>1) %Comienzo a entrenar al modelo. Esta linea de codigo me permite eliminar la foto número 1 de cada clase, ya que la foto número 1 sera usada para testing. Por otro lado mi training estara compuesto por las fotos 2,3,4,5,6,7
        contador1 = contador1 +1; 
        Xtrain_impar(contador1,:) = x_vect; %Almaceno las features en la matriz Xtrain compuesta por 50 clases y cada clase posee 6 fotos, lo que entrega un total de 300 filas y 944 columnas.
        Ytrain_impar(contador1,1) = fotos_impares(k).label; %Almaceno los labels en la matriz Ytrain. 
    else %este else me servirá para formar mi Xtest. 
        contador2 = contador2+1;
        Xtest_impar(contador2,:) = x_vect;%Almaceno las features en la matriz Xtest compuesta por 50 clases y cada clase posee 1 foto, lo que entrega un total de 50 filas y 944 columnas.
        Ytest_impar(contador2,1) = fotos_impares(k).label;%Almaceno los labels correspondients en la matriz Ytest. 
    end
end
%A continuacion se realiza la normalizacion 
[Xtrain_impar,a,b] = Bft_norm(Xtrain_impar,1);
Xtest_impar = Xtest_impar.*(ones(contador2,1)*a) + ones(contador2,1)*b;
%%Notar que es necesario que los labels en Y_train e Ytest tienen que ser
%%de la forma [1,2,3,4,...]. La funcion Bfs_sfs presenta complicaciones en
%%caso de que si Y_train posea una secuencia tal como [1,3,5,7,..] como
%%ocurre en este caso de la tarea. Esto se debe a que SFS no funciona cuando hay clases no consecutivas
%%Por lo mismo se procede a transformar por ejemplo (puede variar dependiendo del orden de lectura), la secuencia
%%de la forma [1,3,5,7,...] a la forma [1,2,3,4,...]
for k= 1:length(Ytrain_impar(:,1))
 Ytrain_impar(k,1)=((Ytrain_impar(k,1)-1)/2)+1; %%funcion matematica para el caso impar que deja las clases de Ytrain_impar de de forma consecutiva 
end
for k= 1:length(Ytest_impar(:,1))
 Ytest_impar(k,1)=((Ytest_impar(k,1)-1)/2)+1;%%funcion matematica para el caso impar que deja las clases de Ytest_impar de forma consecutiva
end
op.m    = 100;                         %Se seleccionaran 100 caracteristicas
op.show = 0;                           % No mostrar los resultados
op.b.name = 'fisher';                  % SFS con Fisher; buscamos maximizar el j
s = Bfs_sfs(Xtrain_impar,double(Ytrain_impar),op); % el output es 's', el cual contiene las 100 features seleccionadas que mejor me separan las 50 clases segun el metodo SFS.
op.k = 1;
Ypred_impar = Bcl_knn_old(Xtrain_impar(:,s),Ytrain_impar,Xtest_impar(:,s),op);%Se utiliza el clasificado KNN con k=1;
disp('Accuracy Parte I:')
Bev_performance(Ypred_impar,Ytest_impar)%me devuelve el accuracy obtenido de la parte I
%%%%%%%%%%%%%%%%%%%%%%%%%PARTE 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off;
M=256; 
N=256;%%M y N son variables para el resize. 
contador1=0;
contador2=0;
%Los comentarios al caso par son equivalentes al caso impar, solo que no se realiza SFS como se explica
%posteriormente.
for k=1:length(fotos_pares)
    fotos_pares(k).addr;
    I = imread(fotos_pares(k).addr);
    I = imresize(I,[N M]);
    x_vect = Bfx_lbp(I,[],opLBP); %944 columnas
    if(fotos_pares(k).numfoto>1)
        contador1 = contador1 +1;
        Xtrain_par(contador1,:) = x_vect;
        Ytrain_par(contador1,1) = fotos_pares(k).label;
    else
        contador2 = contador2+1;
        Xtest_par(contador2,:) = x_vect; 
        Ytest_par(contador2,1) = fotos_pares(k).label;
    end
end
[Xtrain_par,a,b] = Bft_norm(Xtrain_par,1);
Xtest_par = Xtest_par.*(ones(contador2,1)*a) + ones(contador2,1)*b;
for k= 1:length(Ytrain_par(:,1))
 Ytrain_par(k,1)=((Ytrain_par(k,1)-2)/2)+1;%%%%funcion matematica para el caso par que deja las clases de Ytrain_par de de forma consecutiva 
end
for k= 1:length(Ytest_par(:,1))
 Ytest_par(k,1)=((Ytest_par(k,1)-2)/2)+1;%%funcion matematica para el caso par que deja las clases de Ytest_par de de forma consecutiva 
end
%Notar que no se vuelve a realizar el algoritmo SFS, dado que se utilizan
%las 100 features extraidas de la primera parte. 
op.k = 1;
Ypred_par = Bcl_knn_old(Xtrain_par(:,s),Ytrain_par,Xtest_par(:,s),op);
disp('Accuracy Parte II:')
Bev_performance(Ypred_par,Ytest_par)%me devuelve el accuracy obtenido de la parte II