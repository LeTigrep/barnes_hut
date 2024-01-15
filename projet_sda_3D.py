from numpy import *
from random import random, seed
from time import time
import numpy as np
from numpy import zeros, array, sqrt
import random
from math import pi, cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

seed(323)

G = 1.0  # constante gravitationelle
THRESHOLD = 1  # Nombre de corps après lesquels subdiviser Quad
MAXDEPTH = 10
THETA = 0.5  # Rapport de précision Barnes-Hut
ETA = 0.5 # Facteur de ramollissement

NUM_CHECKS = 2  # Compteur

# pour créer l'axe x y z en 3d 
def draw_bbox_matplotlib(node):
    if node is not None:
        min_point = node.bbox.min()
        max_point = node.bbox.max()
        corners = [
            [min_point[0], min_point[1], min_point[2]],
            [max_point[0], min_point[1], min_point[2]],
            [max_point[0], max_point[1], min_point[2]],
            [min_point[0], max_point[1], min_point[2]],
            [min_point[0], min_point[1], max_point[2]],
            [max_point[0], min_point[1], max_point[2]],
            [max_point[0], max_point[1], max_point[2]],
            [min_point[0], max_point[1], max_point[2]],
        ]

        # Définir les sommets du cadre pour la delimitation
        vertices = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]], 
            [corners[0], corners[1], corners[5], corners[4]], 
            [corners[2], corners[3], corners[7], corners[6]], 
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Dessinez le cadre de délimitation sous forme de wireframe
        for vert in vertices:
            poly3d = [[vert[0]], [vert[1]], [vert[2]], [vert[3]]]
            ax.add_collection3d(Poly3DCollection(poly3d, edgecolor=(0, 0, 1), linewidths=0.5, alpha=0.2))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

        # dessine les boites englobantes des enfants de l'octant
        for child in node.children:
            draw_bbox_matplotlib(child)


# fonction qui affiche les corps 
def draw_bodies_matplotlib(POS, MASS):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        if BOUNDS.inside(POS[i]):
            x, y, z = POS[i]
            radius = MASS[i] * 2
            ax.scatter(x, y, z, s=radius, c='black')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()





class Octree:
    """ Class pour N points stockés sous forme d'Octree 3D """
    root = None

    def __init__(self, bbox, N, theta=THETA):
        self.bbox = bbox
        self.N = N
        self.theta = theta
        self.reset()

    # Réinitialise l'octree pour créer par la suite un nouvel octree
    def reset(self):
        self.root = Octant(self.bbox)

    #  Génère les corps initiaux pour la simulation.
    def generate(self):
        self.reset()
        for x in range(self.N):
            self.root.addBody(x, 0)
            
    # Met à jour le système en calculant les accélérations et en mettant à jour les positions et les vitesses des corps.
    def updateSys(self, dt):
        self.calculateBodyAccels()
        global VEL, POS
        VEL += ACC * dt
        POS += VEL * dt

    # calcule les accélérations de tous les corps du système en utilisant la méthode de l'octre
    def calculateBodyAccels(self):
        for k in range(self.N):
            ACC[k] = self.calculateBodyAccel(k)

    # Calcule l'accélération d'un corps en utilisant la méthode de l'octree
    # bodI indice du corps 
    def calculateBodyAccel(self, bodI):
        return self.calculateBodyAccelR(bodI, self.root)

    # Calcule récursivement l'accélération d'un corps en parcourant l'octree
    def calculateBodyAccelR(self, bodI, node):
        acc = zeros(3, dtype=float)
        if node.leaf:
            for k in node.bods:
                if k != bodI:
                    acc += getForce(POS[bodI], 1.0, POS[k], MASS[k])
        else:
            s = max(node.bbox.sideLength)
            d = node.center - POS[bodI]
            r = sqrt(sum(d ** 2))
            if r > 0 and s / r < self.theta:
                acc += getForce(POS[bodI], 1.0, node.com, node.mass)
            else:
                for k in range(8):
                    if node.children[k] is not None:
                        acc += self.calculateBodyAccelR(bodI, node.children[k])
        return acc

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.root.__str__()

#  Calcule la force de gravité entre deux corps
def getForce(p1, m1, p2, m2):
    global NUM_CHECKS
    d = p2 - p1
    r = sqrt(sum(d ** 2)) + ETA
    f = d * G * m1 * m2 / r ** 3
    NUM_CHECKS += 1
    return f





class Octant:
    """Une région 3D de l'espace, contient des corps ponctuels"""
    def __init__(self, bbox, bod=None, depth=0):
        self.bbox = bbox
        self.center = bbox.center
        self.leaf = True  #parent ou non
        self.depth = depth
        self.position = (bbox.min() + bbox.max()) / 2  # Position du cube
        if bod is not None:
            self.setToBody(bod)
            self.N = 1
        else:
            self.bods = []
            self.mass = 0.0
            self.com = array([0.0, 0.0, 0.0], dtype=float)
            self.N = 0

        self.children = [None] * 8  # 8 fils pour le octree

    # Cette méthode est utilisée pour ajouter un corps ponctuel dans l'octant
    def addBody(self, idx, depth):
        if len(self.bods) > 0 or not self.leaf:   # Vérifie si l'octant actuel contient déjà des corps
            if depth >= MAXDEPTH:
                self.bods.append(idx)
            else:
                subBods = [idx]                    # Si la profondeur maximale n'est pas atteinte, divise l'octant en huit sous-octants
                if len(self.bods) > 0:
                    subBods.append(self.bods[0])
                    self.bods = []
                
                 # Parcourt chaque corps à ajouter dans les sous-octants
                for bod in subBods:
                    octantIdx = self.getOctantIndex(bod)
                    if self.children[octantIdx]:
                        self.children[octantIdx].addBody(bod, depth + 1)
                    else:
                        subBBox = self.bbox.getSubOctant(octantIdx)
                        self.children[octantIdx] = Octant(subBBox, bod, depth + 1)

                
                # L'octant actuel n'est plus considéré comme une feuille car il contient des sous-octants
                self.leaf = False            

            self.com = (self.com * self.mass + POS[idx] * MASS[idx]) / (self.mass + MASS[idx])
            self.mass += MASS[idx]
        else:
            self.setToBody(idx)

        self.N += 1

    # maj le centre de masse
    def updateCOM(self):
        if self.leaf:
            self.mass = sum(MASS[x] for x in self.bods)
            self.com = sum(POS[x] * MASS[x] for x in self.bods) / self.mass
        else:
            self.mass = sum(child.mass if child else 0 for child in self.children)
            self.com = sum((child.mass * child.com if child else zeros(3)) for child in self.children) / self.mass

    # elle va contenir un une corps ponctuel 
    def setToBody(self, idx):
        self.bods = [idx]
        self.mass = float(MASS[idx])
        self.com = POS[idx].copy()

    # utilisée pour déterminer dans quel sous-octant de l'octant actuel un corps ponctuel doit être placé
    def getOctantIndex(self, idx):
        return self.bbox.getOctantIdx(POS[idx])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        bodstring = str(self.bods) if len(self.bods) > 0 else "PARENT"
        childCount = "C:%g," % sum(1 for x in self.children if x) if any(self.children) else ""
        childStr = "\n".join(
            "-" * (self.depth + 1) + str(x + 1) + " " + str(self.children[x])
            if self.children[x] else ""
            for x in range(8)
        )
        return f"D{self.depth}{{N:{self.N},M:{round(self.mass, 2)},{childCount}COM:{self.com.round(2)},B:{bodstring}}}{childStr}"
