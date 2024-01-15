from numpy import *
from random import random,seed
from time import time
import pygame
import numpy as np
from numpy import zeros, array, sqrt
import random
from math import pi, cos, sin

pygame.init()
screen_size = (1000, 800)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("N-Body Simulation")
clock = pygame.time.Clock()

seed(323)

G = 1.0 
THRESHOLD = 1 
MAXDEPTH = 10
THETA = 0.5 
ETA = 0.5

NUM_CHECKS = 0

class QuadTree:
    """  classe pour N points stockés sous forme de Quadtree 2D """
    root = None
    def __init__(self, bbox, N, theta = THETA):
        self.bbox = bbox
        self.N = N
        self.theta = theta
        self.root = Quad(bbox)
     
    

    def find_quad_of_body(self, body_idx, node):
        """ Trouver le quad qui contient le corps avec l'index donné. """
        if node is None:
            return None

        if body_idx in node.bods:
            return node

        if not node.leaf:
      
            for child in node.children:
                if child is not None and child.bbox.inside(POS[body_idx]):
                    return self.find_quad_of_body(body_idx, child)
        return None

    def has_moved_significantly(self, body_idx, old_quad):

        if old_quad is None:
            return True  # Considérez-le comme considérablement déplacé
        return not old_quad.bbox.inside(POS[body_idx])

    def find_new_quad(self, body_idx, start_node):

        if start_node.bbox.inside(POS[body_idx]):
            if start_node.leaf:
                return start_node
            else:
                for child in start_node.children:
                    if child and child.bbox.inside(POS[body_idx]):
                        return self.find_new_quad(body_idx, child)
        else:
            if start_node == self.root:
                return None 


    def move_body(self, body_idx, old_quad, new_quad):

        old_quad.remove_body(body_idx)
        

        old_quad.update_mass_and_com(body_idx)


        if old_quad.is_empty() and old_quad != self.root:
            old_quad.collapse()
        
        # Ajouter le corps au nouveau quad
        new_quad.addBody(body_idx, new_quad.depth)

        # Update center of mass and mass for the new quad after addition
        new_quad.update_mass_and_com(body_idx)

    def update_tree(self):
        for body_idx in range(N):
            old_quad = self.find_quad_of_body(body_idx, self.root)

            if old_quad is None:
               # Le corps est en dehors du quad racine, pour l'instant ignorer
                continue

            if self.has_moved_significantly(body_idx, old_quad):
                new_quad = self.find_new_quad(body_idx, self.root)
                print(new_quad)
                if new_quad:
                    self.move_body(body_idx, old_quad, new_quad)


        
    def reset(self):
        self.root = None


    def updateSys(self, dt):
        self.calculateBodyAccels()
        global VEL
        VEL += ACC * dt

    def calculateBodyAccels(self):
        # Mettre à jour la table ACC en fonction du POS actuel
        for k in range(self.N):
            # Check if the body is within the root quad
            if self.find_quad_of_body(k, self.root) is not None:
                ACC[k] = self.calculateBodyAccel(k)
            else:
                # Si le corps est en dehors du quad racine, réglez son accélération à zéro
                ACC[k] = zeros(2, dtype=float)
    def calculateBodyAccel(self, bodI):
        return self.calculateBodyAccelR(bodI, self.root)

    def calculateBodyAccelR(self, bodI, node):
        # Calculer l'accélération sur le corps I
        # La principale différence est que le corps est ignoré dans les calculs
        if node is None:
            return zeros(2, dtype=float)
        acc = zeros(2,dtype=float)
        if (node.leaf):
            # print "Leaf"
          # Nœud feuille, pas d'enfant
            for k in node.bods:
                if k != bodI: # laisse le même corps
                    acc += getForce( POS[bodI] ,1.0,POS[k],MASS[k])
        else:
            s = max( node.bbox.sideLength )
            d = node.center - POS[bodI]
            r = sqrt(d.dot(d))
            # print "s/r = %g/%g = %g" % (s,r,s/r)
            if (r > 0 and s/r < self.theta):
                # Assez loin pour faire une approximation
                acc += getForce( POS[bodI] ,1.0, node.com, node.mass)
            else:
               # Trop proche de l'approximation, récursion vers le bas de l'arbre
                for k in range(4):
                    if node.children[k] != None:
                        acc += self.calculateBodyAccelR(bodI, node.children[k])
        # print "ACC : %s" % acc
        return acc

        

def getForce(p1,m1,p2,m2):
    global NUM_CHECKS
    d = p2-p1
    r = sqrt(d.dot(d)) + ETA
    f = array( d * G*m1*m2 / r**3 )
    NUM_CHECKS += 1
    return f
