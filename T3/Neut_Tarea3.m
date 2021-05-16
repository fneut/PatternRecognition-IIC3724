%Index: lineas 5 a 102 FACES con SFS Y PCA
%lineas 102 a 188 FACES exclusivamente con PCA
%lineas 190 a 300 TORTILLAS 
%lineas 301 a 412 GENDER
%%
%********************FACES CON SFS Y PCA**********************************
clc;
clear all;
close all;
X=load('set05-face-detection.mat');%Se procede a cargar en una variable X el conjunto de datos de faces


% Data Selection: Primer 80% Training Restante 20% Testing
%         > Training (Xtrain1): 211 x 1589
%         > Testing (Xtest1) :  53 x 1589

% Para este dataset se cuenta con 2 clases. Se procede a encontrar a
% efectuar la seleccion de data, siendo el 80% utilizado para training, y
% el 20% utilizado para testing. 
%Se decidió utilizar el código del profesor para esta parte, dado que
%estaba más ordenado que el realizado en la actividad, y se prefirió
%dedicar el tiempo a todo el análisis de clasificadores y selección de
%features, que a ordenar el código de la actividad.

i1 = find(X.d==1);n1=length(i1);m1=round(0.8*n1); %se procede a buscar las filas en las que se tiene el label 1
i2 = find(X.d==2);n2=length(i2);m2=round(0.8*n2); %idem 

Xtrain1 = [X.f(i1(1:m1),:)   ;X.f(i2(1:m2),:)   ];%features de la matriz Xtrain1 para training


Xtest1  = [X.f(i1(m1+1:n1),:);X.f(i2(m2+1:n2),:)];%features de la matriz Xtest1 para testing

dtrain = [ones(m1,1);2*ones(m2,1)];%matriz de labels para Xtrain1

dtest  = [ones(n1-m1,1);2*ones(n2-m2,1)];%matriz de labels para Xtest2

% *** DEFINCION DE DATOS PARA EL TRAINING ***


% Training: Clean --> Permite eliminar features constantes y
% correlacionadas
%         > Training (Xtrain2): 211 x 380 
s_clean   = Bfs_clean(Xtrain1);%se forma la nueva matriz Xtrain2 en base a las features extraidas por clean
Xtrain2   = Xtrain1(:,s_clean);%se puede ver las features extraidas por clean


%Training: Normalizacion
%         > Training (Xtrain3): 211 x 380 
[Xtrain3, a, b] = Bft_norm(Xtrain2,0);%Se normaliza y se guardan valores "a" y "b" de la normalizacion
%los cuales seran usados posteriormente para la normalizacion de testing.

%Training: SFS
%         > Training (Xtrain4): 211 x 250 
%op.m      = 250; %Se seleccionan 250 features
%op.show   = 0; %no mostrar resultados
%op.b.name = 'fisher'; % SFS with Fisher
%s_sfs     = Bfs_sfs(Xtrain3,dtrain,op);
%save('faces','s_sfs')

load ('faces','s_sfs')
Features_after_sfs = X.fn(s_sfs,:);


Xtrain4   = Xtrain3(:,s_sfs);


% Paso 5-Training: PCA
lambda_energy = 10; %Se utilizaran los 10 lambda mas significativos. Despues del 10, los lambda son practicamente despreciables.
[Xtrain5,lambda,A,Xs,mx] = Bft_pca(Xtrain4,lambda_energy);% se procede a realizar la transformacion lineal
%figure(1);bar(lambda/lambda(1))%Se puede graficar los lambdas normalizados. Se aprecia que efectivamente estan ordenados de mayor a menor
%figure(2);imshow([Xtrain4 Xs],[])% 

%Bio_plotfeatures(Xtrain5,dtrain) % Este comando fue muy usado para analizar que tan mezcladas estan las clases para las features
%seleccionadas, lo cual tenia un gran peso en la seleccion del
%clasificador. Nota que para su uso, lambda_energy debe ser menor a 8.

% *** DEFINCION DE DATOS PARA EL TESTING ***


% Paso 2-Testing: clean
Xtest2 = Xtest1(:,s_clean);

%Testing: normalizacion
N = size(Xtest2,1);
Xtest3 = Xtest2.*(ones(N,1)*a) + ones(N,1)*b;%Se procede a normalizar el testing en base a los valores obtenidos en training


%Testing: SFS
Xtest4 = Xtest3(:,s_sfs);%Se seleccionan las features que fueron elegidas por SFS en training

%Testing: PCA
N = size(Xtest4,1);
Xtest5 = (Xtest4 - ones(N,1)*mx)*A(:,1:lambda_energy);% Obtencion de la matriz Y a partir de PCA en training


% *** CLASIFICADOR Y OBTENCION DE ACCURACY***
op1.k = 1;
op1.p=[];
dpred = Bcl_qda(Xtrain5,dtrain,Xtest5,op1); %se utiliza clasificador QDA
accuracy_faces= Bev_performance(dpred,dtest)
display('Accuracy con SFS y PCA');
%%
%********************FACES EXCLUSIVAMENTE CON PCA**********************************
clc;
clear all;
close all;
X=load('set05-face-detection.mat');%Se procede a cargar en una variable X el conjunto de datos de faces


% Data Selection: Primer 80% Training Restante 20% Testing
%         > Training (Xtrain1): 211 x 1589
%         > Testing (Xtest1) :  53 x 1589

% Para este dataset se cuenta con 2 clases. Se procede a encontrar a
% efectuar la seleccion de data, siendo el 80% utilizado para training, y
% el 20% utilizado para testing. 
%Se decidió utilizar el código del profesor para esta parte, dado que
%estaba más ordenado que el realizado en la actividad, y se prefirió
%dedicar el tiempo a todo el análisis de clasificadores y selección de
%features, que a ordenar el código de la actividad.

i1 = find(X.d==1);n1=length(i1);m1=round(0.8*n1); %se procede a buscar las filas en las que se tiene el label 1
i2 = find(X.d==2);n2=length(i2);m2=round(0.8*n2); %idem 

Xtrain1 = [X.f(i1(1:m1),:)   ;X.f(i2(1:m2),:)   ];%features de la matriz Xtrain1 para training


Xtest1  = [X.f(i1(m1+1:n1),:);X.f(i2(m2+1:n2),:)];%features de la matriz Xtest1 para testing

dtrain = [ones(m1,1);2*ones(m2,1)];%matriz de labels para Xtrain1

dtest  = [ones(n1-m1,1);2*ones(n2-m2,1)];%matriz de labels para Xtest2

% *** DEFINCION DE DATOS PARA EL TRAINING ***


% Training: Clean --> Permite eliminar features constantes y
% correlacionadas
%         > Training (Xtrain2): 211 x 380 
s_clean   = Bfs_clean(Xtrain1);%se forma la nueva matriz Xtrain2 en base a las features extraidas por clean
Xtrain2   = Xtrain1(:,s_clean);%se puede ver las features extraidas por clean


%Training: Normalizacion
%         > Training (Xtrain3): 211 x 380 
[Xtrain3, a, b] = Bft_norm(Xtrain2,0);%Se normaliza y se guardan valores "a" y "b" de la normalizacion
%los cuales seran usados posteriormente para la normalizacion de testing.


Xtrain4   = Xtrain3;

% Paso 5-Training: PCA
lambda_energy = 10; %Se utilizaran los 10 lambda mas significativos. Despues del 10, los lambda son practicamente despreciables.
[Xtrain5,lambda,A,Xs,mx] = Bft_pca(Xtrain4,lambda_energy);% se procede a realizar la transformacion lineal
%figure(1);bar(lambda/lambda(1))%Se puede graficar los lambdas normalizados. Se aprecia que efectivamente estan ordenados de mayor a menor
%figure(2);imshow([Xtrain4 Xs],[])% 

%Bio_plotfeatures(Xtrain5,dtrain) % Este comando fue muy usado para analizar que tan mezcladas estan las clases para las features
%seleccionadas, lo cual tenia un gran peso en la seleccion del
%clasificador. Nota que para su uso, lambda_energy debe ser menor a 8.


% *** DEFINCION DE DATOS PARA EL TESTING ***


%Testing: clean
Xtest2 = Xtest1(:,s_clean);

%Testing: normalizacion
N = size(Xtest2,1);
Xtest3 = Xtest2.*(ones(N,1)*a) + ones(N,1)*b;%Se procede a normalizar el testing en base a los valores obtenidos en training


Xtest4 = Xtest3;

%Testing: PCA
N = size(Xtest4,1);
Xtest5 = (Xtest4 - ones(N,1)*mx)*A(:,1:lambda_energy);% Obtencion de la matriz Y a partir de PCA en training



% *** CLASIFICADOR Y OBTENCION DE ACCURACY***
op1.k = 1;
op1.p=[];
dpred = Bcl_qda(Xtrain5,dtrain,Xtest5,op1); %se utiliza clasificador QDA
accuracy_faces= Bev_performance(dpred,dtest)
display('faces exclusivamente con PCA');

%%
%********************TORTILLAS**********************************

clc;
clear all;
close all;
X = load('set04-tortillas.mat'); %Se procede a cargar en una variable X el conjunto de datos de tortillas


%**************SE REMUEVEN FEATURES DE POSICION PARA EVITAR FALSA CORRELACION 
Index_center = find(contains(X.fn,'center of grav'));
Index_ellipse = find(contains(X.fn,'Ellipse-centre'));
for k=1:length(Index_ellipse)
   X.f(:,Index_ellipse(k))=[];
   X.fn(Index_ellipse(k),:)=[];
end
for k=1:length(Index_center)
    X.f(:,Index_center(k))=[];
    X.fn(Index_center(k),:)=[];
end
%*******************************************************************


% Data Selection: Primer 80% Training Restante 20% Testing
%         > Training (Xtrain1): 240 x 1639
%         > Testing (Xtest1) :  60 x 1639

% Para este dataset se cuenta con 3 clases. Se procede a encontrar a
% efectuar la seleccion de data, siendo el 80% utilizado para training, y
% el 20% utilizado para testing. 
%Se decidió utilizar el código del profesor para esta parte, dado que
%estaba más ordenado que el realizado en la actividad, y se prefirió
%dedicar el tiempo a todo el análisis de clasificadores y selección de
%features, que a ordenar el código de la actividad.

i1 = find(X.d==1);n1=length(i1);m1=round(0.8*n1); %se procede a buscar las filas en las que se tiene el label 1
i2 = find(X.d==2);n2=length(i2);m2=round(0.8*n2); %idem 
i3 = find(X.d==3);n3=length(i3);m3=round(0.8*n3); %idem 


Xtrain1 = [X.f(i1(1:m1),:)   ;X.f(i2(1:m2),:);  X.f(i3(1:m3),:)  ]; %features de la matriz Xtrain1 para training

Xtest1  = [X.f(i1(m1+1:n1),:);X.f(i2(m2+1:n2),:);X.f(i3(m3+1:n3),:)]; %features de la matriz Xtest1 para testing

dtrain = [ones(m1,1);2*ones(m2,1);3*ones(m3,1)]; %matriz de labels para Xtrain1

dtest  = [ones(n1-m1,1);2*ones(n2-m2,1);3*ones(n3-m3,1)]; %matriz de labels para Xtest1

% *** DEFINICION DE DATOS PARA EL TRAINING ***


% Training: Clean --> Permite eliminar features constantes y
% correlacionadas
%         > Training (Xtrain2): 240 x 332 
s_clean   = Bfs_clean(Xtrain1);
Features_after_clean =X.fn(s_clean,:);%se forma la nueva matriz Xtrain2 en base a las features extraidas por clean
Xtrain2   = Xtrain1(:,s_clean); %se puede ver las features extraidas por clean


%Training: Normalizacion
%         > Training (Xtrain3): 240 x 332 
[Xtrain3, a, b] = Bft_norm(Xtrain2,0);%Se normaliza y se guardan valores "a" y "b" de la normalizacion
%los cuales seran usados posteriormente para la normalizacion de testing.



%Training: SFS
%         > Training (Xtrain4): 240 x 40 
op.m      = 40;% Se seleccionan 40 features que entregaban gran separabilidad para el dataset dado
op.show   = 0;% no mostrar resultados
op.b.name = 'fisher';% Se utiliza SFS con Fisher como metodo de seleccion de caracteristicas
s_sfs     = Bfs_sfs(Xtrain3,dtrain,op);
Features_after_sfs = X.fn(s_sfs,:); %Se procede a hacer un analisis de las features extraidas para evitar falsa correlacion y redundancia
Xtrain4   = Xtrain3(:,s_sfs);%se seleccionan las nuevas features extraidas por SFS con KNN y se guardan en una nueva matriz Xtrain4



% %Training: Transformacion PCA
lambda_energy = 8; %Se utilizaran los 8 lambda mas significativos. Despues del 8 los lambda son practicamente despreciables.
[Xtrain5,lambda,A,Xs,mx] = Bft_pca(Xtrain4,lambda_energy); % se procede a realizar la transformacion lineal
%figure(1);bar(lambda/lambda(1)) %Se puede graficar los lambdas normalizados. Se aprecia que efectivamente estan ordenados de mayor a menor
%figure(2);imshow([Xtrain4 Xs],[])



%Bio_plotfeatures(Xtrain5,dtrain) % Este comando fue muy usado para analizar que tan mezcladas estan las clases para las features
%seleccionadas, lo cual tenia un gran peso en la seleccion del clasificador



% *** DEFINCION DE DATOS PARA EL TESTING ***

%Testing: clean
Xtest2 = Xtest1(:,s_clean); %Se procede a aplicar las features extraidas por clean en training

%Testing: normalizacion
N = size(Xtest2,1);
Xtest3 = Xtest2.*(ones(N,1)*a) + ones(N,1)*b;%Se procede a normalizar el testing en base a los valores obtenidos en training

%Testing: SFS
Xtest4 = Xtest3(:,s_sfs); %Se seleccionan las features que fueron elegidas por SFS en training

%Testing: PCA
N = size(Xtest4,1);
Xtest5 = (Xtest4 - ones(N,1)*mx)*A(:,1:lambda_energy);% Obtencion de la matriz Y a partir de PCA en training


% *** CLASIFICADOR Y OBTENCION DE ACCURACY***
op1.k = 1;
dpred = Bcl_knn(Xtrain5,dtrain,Xtest5,op1); %se utiliza clasificador knn con k=1
accuracy_tortillas = Bev_performance(dpred,dtest)

%%

%********************GENDER**********************************

clc;
clear all;
close all;
X=load('set06-gender.mat'); %Se procede a cargar en una variable X el conjunto de datos de gender


% ***Se procedió a analizar la feature Gabor, la cual entregaba una buena separabilidad de clases*****
% ***Simplemente se deja prueba de que fue realizado***

% Index_gabor = find(~contains(X.fn,'Gabor'))
% colToDelete = [];
% for k=1:length(Index_gabor)
%    colToDelete = [colToDelete,Index_gabor(k)];
% end
% X.fn(colToDelete,:)=[];
% X.f(:,colToDelete) = [];



% Data Selection: Primer 80% Training Restante 20% Testing
%         > Training (Xtrain1): 488 x 1589
%         > Testing (Xtest1) :  122 x 1589


% Para este dataset se cuenta con 2 clases desordenadas. No obstante Se procede a encontrar a
% efectuar la seleccion de data, siendo el 80% utilizado para training, y
% el 20% utilizado para testing. 
%Se decidió utilizar el código del profesor para esta parte, dado que
%estaba más ordenado que el realizado en la actividad, y se prefirió
%dedicar el tiempo a todo el análisis de clasificadores y selección de
%features, que a ordenar el código de la actividad efectudo.

i1 = find(X.d==1);n1=length(i1);m1=round(0.8*n1);%se procede a buscar las filas en las que se tiene el label 1
i2 = find(X.d==2);n2=length(i2);m2=round(0.8*n2);%idem 


Xtrain1 = [X.f(i1(1:m1),:)   ;X.f(i2(1:m2),:)   ]; %features de la matriz Xtrain1 para training

Xtest1  = [X.f(i1(m1+1:n1),:);X.f(i2(m2+1:n2),:)]; %features de la matriz Xtest1 para testing

dtrain = [ones(m1,1);2*ones(m2,1)]; %labels para el training

dtest  = [ones(n1-m1,1);2*ones(n2-m2,1)]; %labels para el testing




% *** DEFINCION DE DATOS PARA EL TRAINING ***


% Training: Clean --> Permite eliminar features constantes y
% correlacionadas
%         > Training (Xtrain2): 488 x 303 
s_clean   = Bfs_clean(Xtrain1); 
Xtrain2   = Xtrain1(:,s_clean); %se forma la nueva matriz Xtrain2 en base a las features extraidas por clean
Features_after_clean = X.fn(s_clean,:); %se puede ver las features extraidas por clean


% Training: Normalizacion
%         > Training (Xtrain3): 488 x 303
[Xtrain3, a, b] = Bft_norm(Xtrain2,0); %Se normaliza y se guardan valores "a" y "b" de la normalizacion
%los cuales seran usados posteriormente para la normalizacion de testing.


% Training: SFS con KNN
%         > Training (Xtrain4): 488 x 4 
op.m      = 4;% Se seleccionaron 4 features que entregaban una gran separabilidad (Garbor y textura)
op.show   = 0;% no mostrar resultados
op.b.name = 'knn';% se utiliza el metodo SFS con clasificador KNN, cuyo computo es mas pesado, pero entrega mejores resultados 
op.b.options.k = 8; %se usa KNN con k=8
s_sfs     = Bfs_sfs(Xtrain3,dtrain,op); %se aplica metodo 
Features_after_sfs = X.fn(s_sfs,:); %Se procede a hacer un analisis de las features extraidas para evitar falsa correlacion y redundancia
Xtrain4   = Xtrain3(:,s_sfs); %se seleccionan las nuevas features extraidas por SFS con KNN y se guardan en una nueva matriz Xtrain4


%Training: Transformacion PCA
lambda_energy = 4; %Se utilizaran los 4 lambda mas significativos 
[Xtrain5,lambda,A,Xs,mx] = Bft_pca(Xtrain4,lambda_energy); % se procede a realizar la transformacion lineal
%figure(1);bar(lambda/lambda(1)) %Se puede graficar los lambdas normalizados. Se aprecia que efectivamente estan ordenados de mayor a menor
%figure(2);imshow([Xtrain4 Xs],[])

%Bio_plotfeatures(Xtrain5,dtrain) % Este comando fue muy usado para analizar que tan mezcladas estan las clases para las features
%seleccionadas, lo cual tenia un gran peso en la seleccion del clasificador



% *** DEFINCION DE DATOS PARA EL TESTING ***

%Testing: clean
Xtest2 = Xtest1(:,s_clean); %Se procede a aplicar las features extraidas por clean en training

%Testing: normalizacion
N = size(Xtest2,1);  
Xtest3 = Xtest2.*(ones(N,1)*a) + ones(N,1)*b;%Se procede a normalizar el testing en base a los valores usados en training

%Testing: SFS
Xtest4 = Xtest3(:,s_sfs); %Se seleccionan las features que fueron elegidas por SFS con KNN en training


%Testing: PCA
N = size(Xtest4,1);
Xtest5 = (Xtest4 - ones(N,1)*mx)*A(:,1:lambda_energy);% Obtención de la matriz Y a partir de PCA en training


% *** CLASIFICADOR Y OBTENCION DE ACCURACY***
op1.k = 1;
dpred = Bcl_knn(Xtrain5,dtrain,Xtest5,op1); %se utiliza clasificador knn con k=1
accuracy_gender = Bev_performance(dpred,dtest)