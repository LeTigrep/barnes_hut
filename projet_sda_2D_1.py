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

    # fonction qui permet de voir si un corps a changé de postion de manière significative par rapport à l'ancien
    def has_moved_significantly(self, body_idx, old_quad):

        if old_quad is None:
            return True  # Considérez-le comme considérablement déplacé
        return not old_quad.bbox.inside(POS[body_idx])

    # permet de trouver le nouveau quadtree où un corps ponctuel devrait être placé
    def find_new_quad(self, body_idx, start_node):
        # vérifie si le noeud de départ est toujours à l'interieur de boite englobante
        if start_node.bbox.inside(POS[body_idx]):
            if start_node.leaf:
                return start_node  # si oui donc c'est le bon noeud à placer 
            else:                                # si non le corps à donc quitter la boite il faut le remplacer 
                for child in start_node.children:
                    if child and child.bbox.inside(POS[body_idx]):
                        return self.find_new_quad(body_idx, child)
        else:
            if start_node == self.root:   # le corps n'est pas de le quadtree
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


class BoundingBox:
    def __init__(self,box,dim=2):
        assert(dim*2 == len(box))
        self.box = array(box,dtype=float)
        self.center = array( [(self.box[2]+self.box[0])/2, (self.box[3]+self.box[1])/2] , dtype=float)
        self.dim = dim
        self.sideLength = self.max() - self.min()
 
    def max(self):
        return self.box[self.dim:]
    def min(self):
        return self.box[:self.dim]
    def inside(self,p):
        # p = [x,y]
        if any(p < self.min()) or any(p > self.max()):
            return False
        else:
            return True
    def getQuadIdx(self,p):
        # y goes up
        # 0 1
        # 2 3
        if p[0] > self.center[0]: # x > mid
            if p[1] > self.center[1]: # y > mid
                return 1
            else:
                return 3
        else:
            if p[1] > self.center[1]: # y > mid
                return 0
            else:
                return 2
    def getSubQuad(self,idx):
        # 0 1
        # 2 3
        # [x  y x2 y2]
        #  0  1  2  3
        b = array([None,None,None,None])
        if idx % 2 == 0:
            # Even #, left half
            b[::2] = [self.box[0], self.center[0]] # x - midx
        else:
            b[::2] = [self.center[0], self.box[2]] # midx - x2
        if idx < 2:
            # Upper half (0 1)
            b[1::2] = [self.center[1], self.box[3]] # midy - y2
        else:
            b[1::2] = [self.box[1], self.center[1]] # y - midy
        return BoundingBox(b,self.dim)




def draw_bodies_pygame():
    for i in range(N):
        if BOUNDS.inside(POS[i]):
            x, y = convert_to_screen_coords(POS[i])
            pygame.draw.circle(screen, (0, 0, 0), (x, y), int(MASS[i] * 2))

def draw_bbox_pygame(node):
    if node is not None:
        x0, y0 = convert_to_screen_coords(node.bbox.min())
        x1, y1 = convert_to_screen_coords(node.bbox.max())
        pygame.draw.rect(screen, (0, 0, 255), (x0, y0, x1 - x0, y1 - y0), 1)
        if node.leaf:
            color = (0, 255, 0)  # Green for leaf nodes
        else:
            color = (255, 0, 0)  # Red for parent nodes
        pygame.draw.rect(screen, color, (x0, y0, x1 - x0, y1 - y0), 1)
        for child in node.children:
            draw_bbox_pygame(child)

def convert_to_screen_coords(p):
    screen_pos = (p - BOUNDS.min()) / (BOUNDS.max() - BOUNDS.min()) * np.array(screen_size)
    return np.trunc(screen_pos).astype(int)
def draw_quadtree(node, surface):
    if node is None:
        return

    top_left = convert_to_screen_coords(node.bbox.min())
    bottom_right = convert_to_screen_coords(node.bbox.max())
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    if node.leaf:
        color = (0, 255, 0)  # Green for leaf nodes
    else:
        color = (255, 0, 0)  # Red for parent nodes

    # Debugging: Print the coordinates and dimensions
    #print(f"Drawing quad: Top Left: {top_left}, Width: {width}, Height: {height}")

    pygame.draw.rect(surface, color, (top_left[0], top_left[1], width, height), 1)

    for child in node.children:
        draw_quadtree(child, surface)
