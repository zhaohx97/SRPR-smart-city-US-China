import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt



def fit_ellipse_to_polygon(points):
    points = np.array(points)
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    axis_lengths = np.sqrt(eigvals) * 2 
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1])) 
    
    return (x_mean, y_mean), axis_lengths[0], axis_lengths[1], angle

def ellipse_points(center, width, height, angle, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    ellipse = np.array([width / 2 * np.cos(theta), height / 2 * np.sin(theta)])
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]]) 
    ellipse_rotated = np.dot(rotation_matrix, ellipse)
    return ellipse_rotated[0] + center[0], ellipse_rotated[1] + center[1]

def gini_coefficient(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumulative_x = np.cumsum(sorted_x)
    gini_index = (2 * np.sum((np.arange(1, n+1) / n) * cumulative_x)) / cumulative_x[-1] - (n + 1) / n
    return gini_index



y_col_list=[
            'Data-Driven Hypothesis Testing',
            'Exploring Social and Behavioral Impacts',
            'Frameworks and Models',
            'Guidance for Policymakers',
            'Identification of Knowledge Gaps',
            'Investigating Long-Term Trends',
            'Multi-Disciplinary Collaboration',
            'New Theoretical Insights',
            'Quantitative and Qualitative Analysis',
            'Modeling and Simulation',
            'Theoretical Framework Development',
            'Understanding Urban Dynamics'
            ]

x_col_list=[
            'Building Practical Solutions',
            'Collaboration with Industry and Government',
            'Commercial Products or Services',
            'Deploying Smart City Infrastructure',
            'Enhancing Urban Efficiency',
            'Functioning Prototypes or Systems',
            'Improved City Services',
            'Iterative Design and Development',
            'Prototyping and Testing',
            'Supporting Scalability and Commercialization',
            'System Integration',
            'Technological Innovation'
            ]

mapping_dict = {
    'Davidson, Tennessee': 'Davidson, TN',
    'Orange, California': 'Orange, CA',
    'Fulton, Georgia': 'Fulton, GA',
    'Story, Iowa': 'Story, IA',
    'Suffolk, Massachusetts': 'Suffolk, MA',
    'King, Washington': 'King, WA',
    'Cook, Illinois': 'Cook, IL',
    'New York, New York': 'New York, NY',
    'Los Angeles, California': 'Los Angeles, CA',
    'Maricopa, Arizona': 'Maricopa, AZ',
    'Harris, Texas': 'Harris, TX',
    'Santa Clara, California': 'Santa Clara, CA',
    'Philadelphia, Pennsylvania': 'Philadelphia, PA',
    'Washtenaw, Michigan':'Washtenaw, MI'
    }