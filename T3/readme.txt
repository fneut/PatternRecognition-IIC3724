Notas a considerar

1) Usa Bcl_knn, NO Bcl_knn_old

1)Para ejecutar el load, se deben poner los archivos en el directorio de trabajo.

2)Se decidi� poner en un solo script los 3 datasets. Sin embargo, su ejecuci�n es por separado. Se debe usar
ctr+enter (o cmd+enter en mac) para ir corriendo cada c�digo por separado. 

3) Para el caso del dataset Faces, se usaron dos m�todos, y se muestra el accuracy para ambos, nada m�s que para demostrar algo que 
encontr� durante el estudio. Uno utiliza exclusivamente PCA, mientras que el otro SFS con PCA, cuyo tiempo de simulaci�n aumenta
en 50 segundos por la elecci�n de un elevado n�mero de  features mediante SFS. Producto de esos 50 segundos extras, se decidi� poner en 
un archivo aparte 'faces.mat' las caracter�sticas ya extra�das, para que as� sea m�s r�pida la correcci�n. 

Ind�ce:
l�neas 5 a 102 FACES con SFS Y PCA (Ejecuci�n ctr+enter o cmd+enter, "par�ndose en dicha parte del script")
l�neas 102 a 188 FACES exclusivamente con PCA (idem)
l�neas 190 a 300 TORTILLAS (idem)
l�neas 301 a 412 GENDER (idem)