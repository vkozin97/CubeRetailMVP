import numpy as np
from pandas import read_csv, DataFrame
import scipy.stats as stats
import os
import json


dataDir = 'data'
productsFile = os.path.join(dataDir, "productsList.txt")
clustersFile = os.path.join(dataDir, "clustersList.txt")
lambdasFile = os.path.join(dataDir, "lambdas.csv")

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
    
    invalidlambdasArgument = Exception("Argument lambdas must be a filepath or an arraytype with shape (len(productsList), len(clusterList))")
    invalidColumnsType = Exception("Column names must be integers")
    
    linker = '\n' + '-'*30 + '\n'
    
    def __init__(self, productsList=None, clustersList=None, lambdas=None, randomSeed=90, richness=1.5):
        
        ### Arguments must be either 
        if type(productsList) is ProductsList:
            self.productsList = productsList
        else:
            self.productsList = ProductsList(productsList)
        if type(clustersList) is ClustersList:
            self.clustersList = clustersList
        else:
            self.clustersList = ClustersList(clustersList)
        
        self.n = len(self.clustersList)
        self.m = len(self.productsList)

        if lambdas is None:
            self.lambdas = np.zeros((self.n, self.m))
        elif type(lambdas) is str:
            self.readlambdasFromFile(lambdas)
        else:
            self.lambdas = np.array(lambdas)
            if self.lambdas.shape != (self.n, self.m):
                raise self.invalidlambdasArgument
        
        self.clientIds = []
        self.hiddenClusters = dict()
        self.clientProducts = dict()
        
        np.random.seed(randomSeed)
        
        self.richness = richness
                
    def readlambdasFromFile(self, filepath):
        df = read_csv(filepath, sep=',', index_col=0)
        self.lambdas_df = df

        self.lambdas = np.zeros((self.n, self.m)) + 0.01
        for clustId in df.index:
            for prodId in df.columns:
                try:
                    self.lambdas[self.clustersList.getIndexById(clustId), self.productsList.getIndexById(int(prodId))] += df[prodId][clustId]
                except ValueError:
                    raise self.invalidColumnsType
                except IndexError:
                    raise IndexError("Some ids in lambdas file are not existing in products or clusters files")
        
        
        
    def savelambdasToFile(self, filepath):
        df = DataFrame(self.lambdas)
        df.columns = [x.productId for x in self.productsList.productsList]
        df.index = [x.clusterId for x in self.clustersList.clustersList]
        df.to_csv(filepath, sep=',')
        
    def __str__(self):
        df = DataFrame(self.lambdas)
        df.columns = [product.name for product in self.productsList.productsList]
        df.index = [cluster.name for cluster in self.clustersList.clustersList]
        return str(self.productsList) + self.linker + str(self.clustersList) + self.linker + str(df)
    
    def __repr__(self):
        df = DataFrame(self.lambdas)
        df.columns = [product.name for product in self.productsList.productsList]
        df.index = [cluster.name for cluster in self.clustersList.clustersList]
        return str(self.productsList) + self.linker + str(self.clustersList) + self.linker + repr(df)
    
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    @staticmethod
    def logLikelihoodOfRealisationToDistribution(realisation, lambdas):
        realisation = np.array(realisation)
        lambdas = np.array(lambdas)
        assert (realisation.shape == lambdas.shape and len(realisation.shape) == 1 and realisation.shape[0] > 0)
        
        lambdas *= realisation.sum() / lambdas.sum()
        
        likelihood = 0
        for i in range(len(realisation)):
            likelihood += np.log(stats.poisson.pmf(realisation[i], lambdas[i]))
        
        return likelihood
    
    def getProductsListAndClusterPrediction(self, clientId, returnLogLikelihoods=False):
        if not clientId in self.hiddenClusters:
            self.clientIds.append(clientId)
            self.hiddenClusters[clientId] = np.random.randint(self.n)
            self.clientProducts[clientId] = np.zeros(self.m, dtype='int64')
        
        d = self.lambdas[self.hiddenClusters[clientId]]
        buyCoefficient = np.random.poisson(self.richness * 100) / 100.
        resVector = [np.random.poisson(buyCoefficient * d[i]) for i in range(len(d))]
        
        check = dict()
        for i in range(self.m):
            if resVector[i] > 0:
                check[self.productsList[i].name] = resVector[i]
                self.clientProducts[clientId][i] += resVector[i]
    
        logLikelihoods = [ProductsGenerator.logLikelihoodOfRealisationToDistribution(self.clientProducts[clientId], self.lambdas[i]) for i in range(self.n)]
        prediction = np.argmax(logLikelihoods)
    
        if returnLogLikelihoods:
            return check, self.clustersList[prediction], logLikelihoods
        else:
            return check, self.clustersList[prediction]



if __name__ == "__main__":
    generator = ProductsGenerator(productsFile, clustersFile, lambdasFile)
    print(generator)