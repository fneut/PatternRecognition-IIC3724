clt;
close all;
clear all;
clearvars;
warning off;
%%A CONTINUACION SE SELECCIONA EL NOMBRE DE LA CARPETA DONDE ESTAN CONTENIDAS LAS IMAGENES
st = 'pedestrians';%carpeta donde se encuentran las fotos

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
estruc_fotos = dir([st '/*.png']);
index    = randperm(numel(estruc_fotos), length(estruc_fotos));%%Necesario para funcionalidad de codigo leyendo en desorden los archivos de la carpeta


%%%%%%%%%%%%%%%%%%%% Definiciones de matrices%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=[];
d=[];


%%%%%%%%%%Condiciones para realizar HOG%%%%%%%%%%%%%%%%%%%%%%
options.nj    = 20;             % 10 x 20 particiones
options.ni    = 10;            
options.B     = 9;              % 9 bins
options.show  = 0;              % Mostrar resultados
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:length(estruc_fotos)
    full_name = estruc_fotos(index(i)).name; 
    %full_name = estruc_fotos(i).name;
    full_name_cut = strsplit(full_name,'.');
    paridad= full_name_cut{1}(1);
    num_foto= full_name_cut{1};
    %directorio = [st '/' estruc_fotos(i).name];
    directorio = [st '/' estruc_fotos(index(i)).name];
    I = imread(directorio);%procedo a 'leer' la imagen contenida en el archivo
    J=rgb2gray(I);%Transformacion de la imagen a escala de grises
    x_vect = Bfx_hog(J,options); %Extraccion de features mediante Hog
    X=[X; x_vect]; %Se concatenan las features a la matriz X
    if(paridad=='P')
       d=[d;0]; %Se decidio que las imagenes de peatones perteneceran a la clase 0.
    elseif(paridad=='N')
       d=[d;1]; %Se decidio que las imagenes de no-peatones perteneceran a la clase 1.
    end
end


s_clean   = Bfs_clean(X);
X1   = X(:,s_clean);%se forma la nueva matriz X1 en base a las features extraidas por clean. Notar que clean 
[X2, a, inter] = Bft_norm(X1,0);%se forma la nueva matriz X2 normalizada



%%%%%%%%%%%%%%%%%%%SFS%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%Descomentar si se quiere probar. Tiene un tiempo de simulación
%%%%%%%%%%cercano a 2 minutos.
% op.m      = 100; %Se seleccionan 100 features
% op.show   = 0; %no mostrar resultados
% op.b.name = 'fisher'; % SFS with Fisher
% s_sfs     = Bfs_sfs(X2,d,op);
% save('features_extracted','s_sfs')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load ('features_extracted','s_sfs') %Se guardaron en una variable adicional las features extraidas por SFS para evitar tiempos de simulacion en la correcion
X3=X2(:,s_sfs); %Se seleccionan las caracteristicas extraidas por SFS.
%Xn = ['x_1';'x_2'];
%Bio_plotfeatures(X3(:,3:4),d,Xn); %Se utilizo para ver como estaban siendo repartidas las clases para las features seleccionadas


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASIFICADORES A UTILIZAR%%%%%%%%%%%%%%
i = 0;
i=i+1;b(i).name = 'libsvm';   b(i).options.kernel = '-t 0';  b(i).options.string = 'svm-lin';                % Lin-SVM
i=i+1;b(i).name = 'lda'   ;   b(i).options.p = [];           b(i).options.string = 'lda';                    % LDA
i=i+1;b(i).name = 'ann';  b(i).options.hidden   = [5 5 6]; 



%%%%%%%%%%%%%%%%OPTIONS PARA CROSSVAL CON 10 CARPETAS%%%%%%%%%%%%%%%%%%%%%
op.strat=1;  op.v = 10; op.show = 0; op.c = 0.90;  %90% intervalo de confianza, estratificado, 10 carpetas.
       


%%%%%%%%%%%%%%%%%%%%%%%%CLASIFICADOR SVM LINEAL%%%%%%%%%%%%%%%%%%%%%%%%%%%
op.b = b(1);
[clas_ideal,clas_estim,p,ci] = crossval_p(X3,d,op);                                    
[desv,prom]=desemp(clas_ideal,clas_estim,'SVM lineal');


 %%%%%%%%%%%%%%%%%%%%%%%%CLASIFICADOR LDA%%%%%%%%%%%%%%%%%%%%%%%%%%%
op.b = b(2);
[clas_ideal,clas_estim,p,ci] = crossval_p(X3,d,op);                                    
[desv,prom]=desemp(clas_ideal,clas_estim,'lda');


%%%%%%%%%%%%%%%%%%%%%%%CLASIFICADOR NEURAL NETWORKS%%%%%%%%%%%%%%%%%%%%%%%%%%%
op.b = b(3);
[clas_ideal,clas_estim,p,ci] = crossval_p(X3,d,op); 
[desv,prom]=desemp(clas_ideal,clas_estim,'redes neuronales');



%%%%%%%%%%%%%%%%%%%%%%%%%%%FUNCIONES A UTILIZAR%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Se crea una funcion llamada desemp(), la cual recibe como 'input' la
%%%%clasificacion ideal (clas_ideal) y la clase estimada (clas_estim) que
%%%%salen como 'output' del cross validation. En base a ellas se calcula la
%%%%media y la desviacion estandar de: 'precision', 'recall' y 'accuracy'
%%%%de las 10 carpetas realizadas en cross-validation. 


function [desv_estandar,promedios] = desemp(clas_ideal,clas_estim,clasificador)
precision=[];
recall=[];
acc = [];
for carpetas=1:size(clas_estim,2) %Se hace un for sobre los resultados obtenidos en las 10 carpetas.
    [T,pc] = Bev_confusion(clas_ideal(:,carpetas),clas_estim(:,carpetas)); %Entrega la matriz de confusion.
    TP=T(1,1); %True positives 
    FN=T(1,2); %False Negatives
    FP=T(2,1); %False Positives
    TN=T(2,2); %True negatives
    precision_it=TP/(TP+FP);  %formula para obtener precision en alguna de las 10 iteraciones
    recall_it=TP/(TP+FN); %formula para obtener recall en alguna de las 10 iteraciones
    precision=[precision precision_it]; %Se concatena los resultados en una matriz
    recall=[recall recall_it];%Se concatena los resultados en una matriz
    acc=[acc pc];%Se concatena los resultados en una matriz
end
strings={'accuracy','recall','precision'};
promedios = [mean(acc),0;mean(recall),0;mean(precision),0]; %Se calcula la media del 'accuracy','recall' y 'precision' respectivamente
desv_estandar = [std(acc),0;std(recall),0;std(precision),0];%Se calcula la desviacion estandar del 'accuracy','recall' y 'precision' respectivamente
promedios=[num2cell(promedios)];
desv_estandar=[ num2cell(desv_estandar)];
for fil=1:size(promedios,1)
    promedios{fil,2}=strings{fil};
    desv_estandar{fil,2}=strings{fil};
end
for m=1:size(promedios,1)
    if(m==1)
        formatSpec = '\n\nPromedio de %1s utilizando clasificador %1s =  %7.4f \n'; 
    else
        formatSpec = 'Promedio de %1s utilizando clasificador %1s =  %7.4f \n';%Se imprimen resultados de promedio
    end
    fprintf(formatSpec,promedios{m,2},clasificador,promedios{m,1}); 
end
for m=1:size(desv_estandar,1)
    formatSpec = 'Desviacion estandar de %1s utilizando clasificador %1s =  %7.4f \n';%Se imprimen resultados de desviacion estandar
    fprintf(formatSpec,desv_estandar{m,2},clasificador,desv_estandar{m,1}); 
end


end


%%%Se crea la funcion crossval_p() para realizar el cross-validation. Esta
%%%funcion esta basada en Bev_crossval() del Toolbox Balu, no obstante tuvo
%%%que ser levemente modificada para obtener la clasificacion estimada por
%%%los clasificadores, la cual es requerida para obtener los valores de
%%%precision y recall, que es hecho en la funcion desemp().

function [clas_ideal,clas_estim,p,ci] = crossval_p(X,d,options)

v        = options.v;
b        = options.b;
show     = options.show;
c        = options.c;

if isfield(options,'strat')
    strat = options.strat;
else
    strat = 0;
end

if (v==1)
    disp('Warning: cross validation with only one group means data training = data test.');
end

if not(exist('show','var'))
    show=1;
end


n = length(b);
N = size(X,1);

dmin = min(d);
dmax = max(d);
nn   = dmin:dmax;

p   = zeros(n,1);
ci  = zeros(n,2);
for k=1:n

    if (v==1)
        XX = X;
        XXt = X;
        dd = d;
        ddt = d;
    elseif (strat==1)
		temporal = cell(v,2);
		for cl=dmin:dmax
			selec = (d==cl);
			XTemp = X(selec,:);
			dTemp = d(selec,:);
			
			numElem = sum(selec);
            
            % Checking if every class has at least v samples, otherwise raise Warning (Sandipan)
            if numElem < v
                warning('Class %d has only %d samples, less than fold value %d!!!\n',dTemp(1,1),numElem,v)
            end
            
			[i,j] = sort(rand(numElem, 1));
			r = floor(numElem/v);
			
			XTemp = XTemp(j,:);
			dTemp = dTemp(j);
			
			for iTemp=1:v
				if iTemp == v
					rango = ((iTemp-1)*r + 1):numElem;
				else
					rango = ((iTemp-1)*r + 1):((iTemp)*r);
				end
				temporal{iTemp,1} = [temporal{iTemp,1};XTemp(rango,:)];
				temporal{iTemp,2} = [temporal{iTemp,2};dTemp(rango)];
			end
		end
		Xr = [];
		dr = [];
		R = zeros(v,2);
		ant = 0;
		for iTemp=1:v
			Xr = [Xr;temporal{iTemp,1}];
			dr = [dr;temporal{iTemp,2}];
			next = size(dr,1);
			R(iTemp,:) = [ant+1 next];
			ant = next;
		end
	else
        rn = rand(N,1);
        [i,j] = sort(rn);

        Xr = X(j,:);
        dr = d(j);

        r = fix(N/v);
        R = zeros(v,2);
        ini = 1;
        for i=1:v-1
            R(i,:) = [ini ini+r-1];
            ini = ini + r;
        end
        R(v,:) = [ini N];
    end
	
    pp = zeros(v,1);
    clas_estim = [];
    clas_ideal = [];
    for i=1:v
        if (v>1)
            XXt = Xr(R(i,1):R(i,2),:);
            ddt = dr(R(i,1):R(i,2),:);
            XX = [];
            dd = [];
            for j=1:v
                if (j~=i)
                    XX = [XX;Xr(R(j,1):R(j,2),:)];
                    dd = [dd;dr(R(j,1):R(j,2),:)];
                end
            end
        end
        [dds,ops] = Bcl_structure(XX,dd,XXt,b(k));
        clas_estim=[clas_estim dds];
        clas_ideal=[clas_ideal ddt];
        pp(i) = Bev_performance(ddt,dds,nn);
    end

    p(k) = mean(pp);
    s = ops.options.string;
    % Confidence Interval

    if (v>1)
        pm     = mean(pp);
        mu    = pm;
        sigma = sqrt(pm*(1-pm)/N);
        t = (1-c)/2;
        if v>20
            z = norminv(1-t);
        else
            z = tinv(1-t,v-1);
        end
        p1 = max(0,mu - z*sigma);
        p2 = min(1,mu + z*sigma);
        ci(k,:) = [p1 p2];

        if show
            fprintf('%3d) %s  %5.2f%% in (%5.2f, %5.2f%%) with CI=%2.0f%% \n',k,s,p(k)*100,p1*100,p2*100,c*100);
        end
    else
        ci(k,:) = [0 0];
    end
end
end