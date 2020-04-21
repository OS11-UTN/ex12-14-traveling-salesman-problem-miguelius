# TSP

## Install

To install you need python3 and pip. We highly recommend the use of virtualenv.

```bash
pip install -r requirments.txt
```

## Use

### ex12.py

To use it just call the script like this:

```
python ex12.py
```

By default it simulates a graph of 6 nodes. You can change it using the environment variable NODOS. For instance you want to see what happens with 10 nodes:

```
NODOS=10 python ex12.py
```

There's also an environment variable called DIMENSIONES that allowes you to try with graphs in more than just two dimmensions. There's not much change in behavour. Just for fun.

### ex13.py

It is called the same way:

```
python ex13.py
```

It shares the same arguments as environment variables an a couple of extra.

### Parameters 

Let's see them:

| Variable           | Type | Description | Default | Supported by |
|---|:---:|---|:---:|:---:|
| SEMILLA            | Integer | Random seed to control | nanosecond | all |
| NODOS              | Integer | Graph's nodes number | 6 | all |
| DIMENSIONES        | Integer | Dimmensions of each point generated. 2 means points in a plane, 3 in a space, and so on.| 2 | all |
| NO_IMPROVMENTS_MAX | Integer | Number iterations without observing and improvment | 2 |  ex13 |
| POPULATION_SIZE    | Integer | Population size for each iteration of the GA | 10 | ex13 |
| ELITE_SIZE         | Integer | Number of elements taken as parents in each generation of GA | 5 | ex13 | 
| MUTATION_RATEO     | Float | Rate of mutation for each generation. The higher the more random. | 0.5 | ex13 | 

