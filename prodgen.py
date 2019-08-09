import numpy as np
from pandas import read_csv, DataFrame
import os
import json


dataDir = 'data'
productsFile = os.path.join(dataDir, "productsList.txt")
clustersFile = os.path.join(dataDir, "clustersList.txt")
weightsFile = os.path.join(dataDir, "weights.csv")

class Product:
    def __init__(self, productId, name, description="", price=None):
        self.productId = productId
        self.name = name
        self.description = description
        self.price = price
        
    def __str__(self):
        ans = 'Product ' + str(self.productId) + ": " + self.name
        if not self.description == "":
            ans += " (" + self.description + ")"
        if not self.price is None:
            ans += ' - Price: ' + str(self.price)
        return ans
    
    def __repr__(self):
        return str(self)
        
        
class ProductsList:
    
    invalidArgument = Exception("Argument productsList must be either filepath or a list of Product objects")
    def dublicatingIndexesException(self, id):
        return Exception("Two objects with same productId=" + str(id) + " is being stored")
    
    def __init__(self, productsList=None):
        if productsList is None:
            self.productsList = []
        elif type(productsList) is str:
            self.readFromFile(productsList)
        elif type(productsList) is list:
            self.productsList = []
            for x in productsList:
                if not (type(x) is Product):
                    raise self.invalidArgument
                self.productsList.append(x)
        else:
            raise self.invalidArgument
            
        self.selfOrganize()
            
    def readFromFile(self, filepath):
        with open(filepath) as f:
            self.productsList = [Product(**x) for x in json.load(f)]
        
    def saveToFile(self, filepath):
        with open(filepath, 'w') as f:
            json.dump([x.__dict__ for x in self.productsList], f, indent=4)
        
    def __getitem__(self, i):
        return self.productsList[i]
        
    def __len__(self):
        return len(self.productsList)
    
    def __str__(self):
        return '\n'.join([str(product) for product in self.productsList])
    
    def __repr__(self):
        return str(self)
    
    def getIndexById(self, id):
        for i in range(len(self.productsList)):
            if self.productsList[i].productId == id:
                return i
        return None
    
    def selfOrganize(self):
        count = dict()
        for product in self.productsList:
            id = product.productId
            if id in count:
                raise self.dublicatingIndexesException(id)
            else:
                count[id] = 1
        
        self.productsList = sorted(self.productsList, key=lambda x: x.productId)
                
        
        
class Cluster:
    def __init__(self, clusterId, name, description=""):
        self.clusterId = clusterId
        self.name = name
        self.description = description
        
    def __str__(self):
        ans = "Cluster " + str(self.clusterId) + ': ' + self.name
        if not self.description == "":
            ans += " (" + self.description + ")"
        return ans
    
    def __repr__(self):
        return str(self)
        
        
class ClustersList:
    
    invalidArgument = Exception("Argument clustersList must be either filepath or a list of Cluster objects")
    def dublicatingIndexesException(self, id):
        return Exception("Two objects with same clusterId=" + str(id) + " is being stored")
    
    def __init__(self, clustersList=None):
        if clustersList is None:
            self.clustersList = []
        elif type(clustersList) is str:
            self.readFromFile(clustersList)
        elif type(clustersList) is list:
            self.clustersList = []
            for x in clustersList:
                if not (type(x) is Cluster):
                    raise self.invalidArgument
                self.clustersList.append(x)
        else:
            raise self.invalidArgument
        
        self.selfOrganize()
    
    def readFromFile(self, filepath):
        with open(filepath) as f:
            self.clustersList = [Cluster(**x) for x in json.load(f)]
        
    def saveToFile(self, filepath):
        with open(filepath, 'w') as f:
            json.dump([x.__dict__ for x in self.clustersList], f, indent=4)
        
    def __getitem__(self, i):
        return self.clustersList[i]
        
    def __len__(self):
        return len(self.clustersList)

    def __str__(self):
        return '\n'.join([str(cluster) for cluster in self.clustersList])
    
    def __repr__(self):
        return str(self)
    
    def getIndexById(self, id):
        for i in range(len(self.clustersList)):
            if self.clustersList[i].clusterId == id:
                return i
        return None
    
    def selfOrganize(self):
        count = dict()
        for cluster in self.clustersList:
            id = cluster.clusterId
            if id in count:
                raise self.dublicatingIndexesException(id)
            else:
                count[id] = 1
        
        self.clustersList = sorted(self.clustersList, key=lambda x: x.clusterId)


class ProductsGenerator:
    
    invalidWeightsArgument = Exception("Argument weights must be a filepath or an arraytype with shape (len(productsList), len(clusterList))")
    invalidColumnsType = Exception("Column names must be integers")
    
    linker = '\n' + '-'*30 + '\n'
    
    def __init__(self, productsList=None, clustersList=None, weights=None, randomSeed=110):
        
        ### Arguments must be either 
        if type(productsList) is ProductsList:
            self.productsList = productsList
        else:
            self.productsList = ProductsList(productsList)
        if type(clustersList) is ClustersList:
            self.clustersList = clustersList
        else:
            self.clustersList = ClustersList(clustersList)
        
        self.n = len(self.productsList)
        self.m = len(self.clustersList)

        if weights is None:
            self.weights = np.zeros((self.n, self.m))
        elif type(weights) is str:
            self.readWeightsFromFile(weights)
        else:
            self.weights = np.array(weights)
            if self.weights.shape != (self.n, self.m):
                raise self.invalidWeightsArgument
                
    def readWeightsFromFile(self, filepath):
        df = read_csv(filepath, sep=',', index_col=0)

        self.weights = np.zeros((self.n, self.m))
        for prodId in df.index:
            for clustId in df.columns:
                try:
                    self.weights[self.productsList.getIndexById(prodId), self.clustersList.getIndexById(int(clustId))] = df[clustId][prodId]
                except ValueError:
                    raise self.invalidColumnsType
        
    def saveWeightsToFile(self, filepath):
        df = DataFrame(self.weights)
        df.index = [x.productId for x in self.productsList.productsList]
        df.columns = [x.clusterId for x in self.clustersList.clustersList]
        df.to_csv(filepath, sep=',')
        
    def __str__(self):
        df = DataFrame(self.weights)
        df.index = [product.name for product in self.productsList.productsList]
        df.columns = [cluster.name for cluster in self.clustersList.clustersList]
        return str(self.productsList) + self.linker + str(self.clustersList) + self.linker + str(df)
    
    def __repr__(self):
        df = DataFrame(self.weights)
        df.index = [product.name for product in self.productsList.productsList]
        df.columns = [cluster.name for cluster in self.clustersList.clustersList]
        return str(self.productsList) + self.linker + str(self.clustersList) + self.linker + repr(df)


if __name__ == "__main__":
    generator = ProductsGenerator(productsFile, clustersFile, weightsFile)
    print(generator)