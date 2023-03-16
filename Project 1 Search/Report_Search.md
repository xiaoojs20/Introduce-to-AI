# Project 1: Search

电02 肖锦松 2020010563

## Question 1~4 : Finding a Fixed Food Dot using DFS, BFS, UCS, A\*

按照通用的Graph Search方法实现：

```
function GRAPH-SEARCH(problem) returns a solution or failure
	initialize the frontier using the initial state of problem
	"initialize the explored set to be empty"
	loop do
		if the frontier is empty then return failure
    	choose a leaf node and remove it from the frontier
        if the node contains a goal state then return the corresponding solution
        "add the node to the explored set"
    	expand the chosen node, adding the resulting nodes to the frontier
    		"only if not in the frontier or explored set"
```

对于DFS，**Frontier is a FILO stack**. 

对于BFS，**Frontier is a FIFO queue**

对于UCS，**Frontier is a PriorityQueue**, sorted by g(n)

对于A\*S，**Frontier is a PriorityQueue**, sorted by f(n)=g(n)+h\*(n)

> ***g(n) is the cost from root to n.*** 
> ***h\*(n) is the optimal cost from n to the closest goal***

这里需要注意的是，该问题中一个State或是Node，其实是一个三元组，包括`(stateName, action, Cost)`，这一点需要根据提示先了解类中一些确认的函数的State格式，我们编写的函数应该要与现有代码保持一致。

在将一个State加入Frontier或Expanded Set之前，应该要对当前State进行判断，如果当前State不在Expanded Set中，才能将其加入。否则可能会陷入搜索的闭环，导致程序出错。

---

**Compare** how these algorithms perform in Pac-Man environment, e.g. state numbers, time, etc

> command:
>
> python pacman.py -l tinyMaze -p SearchAgent
> python pacman.py -l mediumMaze -p SearchAgent
> python pacman.py -l bigMaze -z .5 -p SearchAgent
>
> python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs
> python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
> python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
>
> python pacman.py -l tinyMaze -p SearchAgent -a fn=ucs
> python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
> python pacman.py -l bigMaze  -p SearchAgent -a fn=ucs -z .5
>
> python pacman.py -l tinyMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
> python pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
> python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

tinyMaze

| Search Algorithms | Cost | Time | expanded nodes | Score |
| ----------------- | ---- | ---- | -------------- | ----- |
| DFS               | 10   | 0.0s | 15             | 500   |
| BFS               | 8    | 0.0s | 15             | 502   |
| UCS               | 8    | 0.0s | 15             | 502   |
| A\*S              | 8    | 0.0s | 14             | 502   |

mediumMaze

| Search Algorithms | Cost | Time | expanded nodes | Score |
| ----------------- | ---- | ---- | -------------- | ----- |
| DFS               | 130  | 0.0s | 146            | 380   |
| BFS               | 68   | 0.0s | 269            | 442   |
| UCS               | 68   | 0.0s | 269            | 442   |
| A\*S              | 68   | 0.0s | 221            | 442   |

bigMaze

| Search Algorithms | Cost | Time | expanded nodes | Score |
| ----------------- | ---- | ---- | -------------- | ----- |
| DFS               | 210  | 0.0s | 390            | 300   |
| BFS               | 210  | 0.0s | 620            | 300   |
| UCS               | 210  | 0.0s | 620            | 300   |
| A\*S              | 210  | 0.0s | 549            | 300   |

从上述3个Maze、4种Search Algorithms的结果来看，可以得到以下结论：

- DFS不一定可以找到最小Cost的路径，因此DFS is not optimal. 
- BFS, UCS, A\*从上述**仅有**的结果来看，**很可能**找到了最小Cost的路径，这也符合理论上的BFS and UCS is optimal. 
- 相对来说，Expanded Nodes数量从多到少依次是：BDF = UCS > A\* > DFS
- 这里由于问题规模并不大，因此无法显式比较出四种算法的时间复杂度与空间复杂度区别，实际上应如下图所示

<img src="Report_Search.assets/image-20230315121025807.png" alt="image-20230315121025807" style="zoom: 80%;" />



## Question 5~6: Finding All the Corners

第一部分是完成 `CornersProblem`内容的编写，这部分可以参照class FoodSearchProblem的模式来完成。

问题的关键在于，将state[0]类似地置为position，将state[1]设置为expandedCorners. 因此在判断是否为目标时可以根据expandedCorners长度是否为4，在寻找下一个状态时不仅需要计算出其position，还需要判断该position是否在Corners，并视之向expandedCorners增加元素。

第二部分是完成`cornersHeuristic`的编写，考虑Heuristic为当前position到最近未探索的Corner的曼哈顿距离，然后依次考虑未探索Corner到未探索Corner的曼哈顿距离。比如当前左上Corner已经访问过，此时position为$(x,y)$，那么此时的Heuristic为$(x,y)$到最近Corner（假如是右上）的曼哈顿距离，再加上右上到右下Corner以及右下到左下Corner的曼哈顿距离。

这里刚开始写的时候使用繁琐的循环，后来进一步采用了列表推导式和匿名函数两个方便的语法。

`remain_corner = [x for x in corners if x not in expanded_corner]`这一行使用了列表推导式来创建一个新的列表`remain_corner`。它遍历`corners`列表中的每个元素`x`，并检查`x`是否不在`expanded_corner`列表中。如果满足条件，则将元素添加到新列表中。

`remain_corner.sort(key = lambda x: util.manhattanDistance(x, position))`这一行定义了一个匿名函数，并将其赋值给变量`key`。这个匿名函数接受一个参数`x`，并返回`util.manhattanDistance(x, position)`的值。然后把`remain_corner`中的元素根据其到`position`的距离排序。

## Question 7: Eating All The Dots

本题和上一题第二部分相似，要解决吃掉所有豆子的问题，先完成`FoodSearchProblem`内容的熟悉，然后完成`foodHeuristic`的编写

注意到`game.py`中Grid类的`asList`可以帮助我们返回food所在position

启发式函数方面：首先计算Pacman所在位置到最近food的曼哈顿距离；然后我们可以采用**最小生成树的**思想来估计剩余food导致的距离，将所有的food当作节点，使用最小生成树算法，得到其长度；这里采用**改进**曼哈顿距离，如果两个food之间存在墙，那么距离+=2

这里还要注意一个很坑的点！一般来说二维list的原点在左上角，但`asList`函数生成的list的原点在左下角，因此应该统一用`asList`函数的思想转换坐标。

## Question 8: Suboptimal Search

本题的Agent策略为迭代贪婪吃掉最近的点。对于Agent我们只需要采用BFD或是UCS算法搜索到距离最近的food即可。在这之前还需要完成`AnyFoodSearchProblem`的`isGoalState`函数填写，其实这个函数就是说明目标状态是Pacman所在position有food。

这一部分的代码量比较少。
