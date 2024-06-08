

class Graph:
    def __init__(self) -> None:
        self.nodes = list()

    def add_node(self,e1,e2,re):
        self.nodes.append((e1.lower(),e2.lower(),re))

    def find_nodes(self,e):
        return_nodes = list()
        for node in self.nodes:
            if(node[0]==e or node[1]==e):
                return_nodes.append(node)
        return return_nodes
    
    def entities_list(self):
        e = set()
        for (e1,e2,_) in self.nodes:
            e.add(e1)
            e.add(e2)
        return e