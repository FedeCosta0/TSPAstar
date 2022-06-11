Windows
1)Aprire cartella progetto con terminale
2)python -m venv venv (comando per creare virtual environment)
3)venv\Scripts\activate.bat (comando per attivare il venv)
4)pip install -r requirements.txt (comando per installare all'interno del venv i packages necessari)
5)python TSP_solver.py --num_cities --map_size  --cities 
e.g. python TSP_solver.py --num_cities 5 --map_size 10 --cities 6,1 4,3 9,0 1,5 3,2
     python TSP_solver.py --num_cities 5 --map_size 10 se il parametro --cities è omesso le città vengono generate randomicamente

Unix/macOS
1)Aprire cartella progetto con terminale
2)python -m venv venv (comando per creare virtual environment)
3)source venv\bin\activate (comando per attivare il venv)
4)pip install -r requirements.txt (comando per installare all'interno del venv i packages necessari)
5)python TSP_solver.py --num_cities --map_size  --cities 
e.g. python TSP_solver.py --num_cities 5 --map_size 10 --cities 6,1 4,3 9,0 1,5 3,2
     python TSP_solver.py --num_cities 5 --map_size 10 se il parametro --cities è omesso le città vengono generate randomicamente


