import numpy as np
import math

"""
# Quaternion
"""
class Quaterniond:
    def __init__(self, x=1.0, y=0.0, z=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.normalized()

    def normalized(self):
        d = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w) 
        self.x = self.x / d
        self.y = self.y / d
        self.z = self.z / d
        self.w = self.w / d

"""
# rotation transformation
"""
def Quaternion2Euler(quat):
    roll = math.atan2(
        2.0 * (quat.w*quat.x + quat.y*quat.z), 
        1.0 - 2.0 * (quat.x*quat.x + quat.y*quat.y)
    )

    sinp = 2.0 * (quat.w*quat.y - quat.z*quat.x)
    sinp = +1.0 if sinp > +1.0 else sinp
    sinp = -1.0 if sinp < -1.0 else sinp
    pitch = math.asin(sinp)

    yaw = math.atan2(
        2.0 * (quat.w*quat.z + quat.x*quat.y), 
        1.0 - 2.0 * (quat.y*quat.y + quat.z*quat.z)
    )
    return np.array([ [roll],[pitch],[yaw] ])

def Quaternion2Rot(quat):
    rot = np.eye(3)

    rot[0,0] = 1 - 2 * (quat.y*quat.y + quat.z*quat.z)
    rot[0,1] = 2 * (quat.x*quat.y - quat.z*quat.w)
    rot[0,2] = 2 * (quat.x*quat.z + quat.y*quat.w)

    rot[1,0] = 2 * (quat.x*quat.y + quat.z*quat.w)
    rot[1,1] = 1 - 2 * (quat.x*quat.x + quat.z*quat.z)
    rot[1,2] = 2 * (quat.y*quat.z - quat.x*quat.w)
    
    rot[2,0] = 2 * (quat.x*quat.z - quat.y*quat.w)
    rot[2,1] = 2 * (quat.y*quat.z + quat.x*quat.w)
    rot[2,2] = 1 - 2 * (quat.x*quat.x + quat.y*quat.y)
    return rot

def Rotation2Quaternion(rot):
    tr  = rot[0,0] + rot[1,1] + rot[2,2]     # trace
    if tr>0:
        s = math.sqrt(1.0 + tr) * 2.0
        w = s * 0.25
        x = (rot[2,1] - rot[1,2]) / s
        y = (rot[0,2] - rot[2,0]) / s
        z = (rot[1,0] - rot[0,1]) / s
    elif  rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]:
        s = math.sqrt(1.0 +  rot[0,0] - rot[1,1] - rot[2,2] ) * 2.0 #  S=4*qx 
        w = (rot[2,1] - rot[1,2]) / s
        x = s * 0.25
        y = (rot[1,0] + rot[0,1]) / s
        z = (rot[0,2] + rot[2,0]) / s
    elif rot[1,1] > rot[2,2]:
        s = math.sqrt(1.0 - rot[0,0] + rot[1,1] - rot[2,2] ) * 2.0 # S=4*qy
        w = (rot[0,2] - rot[2,0]) / s
        x = (rot[1,0] + rot[0,1]) / s
        y = s * 0.25
        z = (rot[2,1] + rot[1,2]) / s
    else:
        s = math.sqrt(1.0 - rot[0,0] - rot[1,1] + rot[2,2] ) * 2.0 # S=4*qz
        w = (rot[1,0] - rot[0,1]) / s
        x = (rot[0,2] + rot[2,0]) / s
        y = (rot[2,1] + rot[1,2]) / s
        z = s * 0.25
    return Quaterniond(x,y,z,w)

def Euler2Quaternion(eulers):
    r = eulers[0,0]
    p = eulers[1,0]
    y = eulers[2,0]

    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return Quaterniond(qx,qy,qz,qw)

def Euler2Rot(euler):
    def C(x):
        return np.cos(x)
    def S(x):
        return np.sin(x)
    
    ii = euler[0,0]
    jj = euler[1,0]
    kk = euler[2,0]
    R = [[C(kk) * C(jj), C(kk) * S(jj) * S(ii) - S(kk) * C(ii), C(kk) * S(jj) * C(ii) + S(kk) * S(ii)],
         [S(kk) * C(jj), S(kk) * S(jj) * S(ii) + C(kk) * C(ii), S(kk) * S(jj) * C(ii) - C(kk) * S(ii)],
         [-S(jj), C(jj) * S(ii), C(jj) * C(ii)]]
    return np.array(R)

# robot to camera
def rotConvert(rot_in):
    transitRot = np.array([
        [ 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [ 0.0,-1.0, 0.0]
    ])
    return rot_in@transitRot
