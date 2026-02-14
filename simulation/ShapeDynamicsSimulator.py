import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

class BaseShapeSimulation:
    g = 9.81 # gravitational acceleration
    a1 = 5.0
    def __init__(self, m, k1, k2, c1, c2, init_conditions, time_steps, shape_params, color, noise_mean=0, noise_std_dev=0.1):
        self.m = m
        self.k1 = k1
        self.k2 = k2
        self.c1 = c1
        self.c2= c2
        self.init_conditions = init_conditions
        self.time_steps = time_steps
        self.shape_params = shape_params
        self.color = color
        self.noise_mean = noise_mean
        self.noise_std_dev = noise_std_dev
        self.solution = None
        self.shape = None

        # Measurement matrix (assuming we measure both position and velocity in both dimensions)
        self.C = np.eye(4)

        # Additional properties to store simulation data
        self.position_data = []
        self.velocity_data = []
        self.acceleration_data = []
        self.position_data_noisy = []
        self.velocity_data_noisy = []
        self.acceleration_data_noisy = []

        self.fig, self.ax = plt.subplots()

        # Set up the plot
        self.setup_axes()
        self.create_shape()  # Create the shape 


    def setup_axes(self):
        # Dynamically set the limits based on initial conditions
        # initial_y = self.init_conditions[0]
        # y_range_low = 25  # Adjust this value based on how much you want to see around the initial position
        # y_range_high = 2
        self.ax.set_xlim((0, 27))
        # self.ax.set_ylim((initial_y - y_range_low, initial_y + y_range_high)) 
        self.ax.set_ylim(-15,20) #fixed the limits to have a fixed transformation between pixel based axis frame and the simulation axis frame
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def differential_equation(self, state, t):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def solve(self):
        # Solve the differential equation
        self.original_solution = odeint(self.differential_equation, self.init_conditions, self.time_steps, args=(self.m, self.k1, self.k2, self.c1, self.c2))

        # Measurement noise
        measurement_noise = np.random.normal(self.noise_mean, self.noise_std_dev, self.original_solution.shape)
        
        # Apply the measurement matrix and add noise
        self.noisy_solution = (self.C @ self.original_solution.T).T + measurement_noise

    def create_shape(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def init_animation(self):
        return self.shape,

    def update_animation(self, frame):
        # Update position and velocity data
        x, vx, y, vy = self.original_solution[frame]
        x_noisy, vx_noisy, y_noisy, vy_noisy = self.noisy_solution[frame]
        

        # self.position_data.append({"x": 0, "y": y, "z": self.shape_params['center_z']})
        self.position_data.append({"x": 0, "y": y, "z": x})
        self.velocity_data.append({"x": 0, "y": vy, "z": vx})
        self.position_data_noisy.append({"x": 0 , "y": y_noisy, "z": x_noisy})
        self.velocity_data_noisy.append({"x": 0, "y": vy_noisy, "z": vx_noisy})


        # Call the subclass-specific implementation
        self._update_shape(frame)
    
    def _update_shape(self, frame):
        # This method should be implemented by subclasses to update the shape
        raise NotImplementedError("This method should be implemented by subclasses.")

    def animate(self, fps=30):
        self.fps = fps
        self.interval_ms = 1000 / self.fps

        self.solve()

        self.anim = FuncAnimation(self.fig, self.update_animation, frames=len(self.time_steps), init_func=self.init_animation, blit=True, interval=self.interval_ms)
        # plt.show()

    def save_animation(self, file_name='animation.mp4'):
        writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)
        self.anim.save(file_name, writer=writer)
        plt.close(self.fig)

class CircleSimulationGravity(BaseShapeSimulation):
    def create_shape(self):
        # self.shape_params['center_z'] = self.init_conditions[0] #z is the horizontal axis
        self.shape_params['center_x'] = self.init_conditions[0]  # Align with initial conditions x position
        self.shape_params['center_y'] = self.init_conditions[2]  # y position
        # Create the shape once and add it to the axes
        self.shape = mpatches.Ellipse((self.shape_params['center_x'], self.shape_params['center_y']), 
                                      self.shape_params['width'], 
                                      self.shape_params['height'], 
                                      facecolor=self.color)
        self.ax.add_patch(self.shape)

    
    def differential_equation(self, state, t, m, k1, k2, c1, c2):
        x, vx, y, vy = state
        ax = (1/m) * 2 # No external horizontal forces
        ay = -self.g + (1/m) * self.a1  # Vertical forces include gravity and a vertical thrust or acceleration
        return [vx, ax, vy, ay]

    
    def update_animation(self, frame):
        x, vx, y, vy = self.original_solution[frame]
        x_noisy, vx_noisy, y_noisy, vy_noisy = self.noisy_solution[frame]

        # Update position and velocity data
        self.position_data.append({"x": 0, "y": y, "z": x})
        self.velocity_data.append({"x": 0, "y": vy, "z": vx})
        self.acceleration_data.append({"x": 0.0, "y": -self.g + (1/self.m) * self.a1 , "z": (1/self.m) * 2})
        self.position_data_noisy.append({"x": 0, "y": y_noisy, "z": x_noisy})
        self.velocity_data_noisy.append({"x": 0, "y": vy_noisy, "z": vx_noisy})
        self.acceleration_data_noisy.append({"x": 0.0, "y": 0, "z": 0.0})

        # Update the shape's position based on noisy data
        self.shape.set_center((x, y))

        # Return the updated shape
        return self.shape,

    def _update_shape(self, frame):
        # Update the position of the shape
        x, vx, y, vy = self.original_solution[frame]
        self.shape.set_center((x, y))


class FancyBoxSimulationGravitySpringDamper(BaseShapeSimulation):
    def create_shape(self):
        default_pad = 0.3
        self.shape_params['center_x'] = self.init_conditions[0]  # Align with initial conditions  Assuming init_conditions[0] is x
        self.shape_params['center_y'] = self.init_conditions[2]  # Assuming init_conditions[2] is y

        self.shape = mpatches.FancyBboxPatch(
            (self.shape_params['center_x'], self.shape_params['center_y']),
            self.shape_params['width'],
            self.shape_params['height'],
            ec="none",
            boxstyle=mpatches.BoxStyle("Round", pad=self.shape_params['pad'])
        )
        self.shape.set_facecolor(self.color)
        self.ax.add_patch(self.shape)


    def differential_equation(self, state, t, m, k1, k2, c1, c2):
        x, vx, y, vy = state
        ax = -k2/m * x - c2/m * vx  # Horizontal dynamics
        ay = k1/m * y - (c1/m) * vy - self.g # Vertical dynamics
        return [vx, ax, vy, ay]


    def update_animation(self, frame):
        x, vx, y, vy = self.original_solution[frame]
        x_noisy, vx_noisy, y_noisy, vy_noisy = self.noisy_solution[frame]

        # Update data for logging
        self.position_data.append({"x": 0, "y": y, "z": x})
        self.velocity_data.append({"x": 0, "y": vy, "z": vx})
        self.acceleration_data.append({"x":0 , "y": -self.k1/self.m * y - self.c1/self.m * vy - self.g, "z": -self.k2/self.m * x - self.c2/self.m * vx})
        self.position_data_noisy.append({"x": 0, "y": y_noisy, "z": x_noisy})
        self.velocity_data_noisy.append({"x": 0, "y": vy_noisy, "z": vx_noisy})
        self.acceleration_data_noisy.append({"x": 0, "y": 0, "z": 0})

        # Update the shape's position based on noisy data
        self.shape.set_bounds((x_noisy - self.shape_params['width'] / 2, y_noisy - self.shape_params['height'] / 2,
                                self.shape_params['width'], self.shape_params['height']))
        return self.shape,

    def _update_shape(self, frame):
        x, vx, y, vy = self.original_solution[frame]
        # new_x_center = self.shape_params['center_z']
        new_x_center = x
        new_y_center = y
        # Calculate new bounds
        new_bounds = (
            new_x_center - self.shape_params['width'] / 2,
            new_y_center - self.shape_params['height'] / 2,
            self.shape_params['width'],
            self.shape_params['height']
        )
        self.shape.set_bounds(new_bounds)

class FancyBoxSimulationGravity(FancyBoxSimulationGravitySpringDamper):
    # def differential_equation(self, state, t, m, k1,k2, c1, c2):
    #     dydt = [state[1], -BaseShapeSimulation.g + BaseShapeSimulation.a1]
    #     return dydt
    
    def differential_equation(self, state, t, m, k1, k2, c1, c2):
        x, vx, y, vy = state
        ax = (1/m) * 2 # No external horizontal forces
        ay = -self.g + (1/m) * self.a1  # Vertical forces include gravity and a vertical thrust or acceleration
        return [vx, ax, vy, ay]


# Example instantiation: #2D motion
# [initial_x, initial_vx, initial_y, initial_vy] = [18, 0, -1, 0]
# init_conditions = [initial_x, initial_vx, initial_y, initial_vy]  # e.g., [0, 0, 15, 0]
# shape_params = {'center_z': 5, 'width': 4.5, 'height': 4.5, 'color': 'green'}

# circle_sim = CircleSimulationGravity(m=1.0, k1=3, k2 = 15, c1=0.3, c2 = 0.7, init_conditions=init_conditions, 
#                                      time_steps=np.linspace(0, 10, 301), shape_params=shape_params, 
#                                      color='green', noise_mean=0, noise_std_dev=0.0)
# circle_sim.animate()
# circle_sim.save_animation('circle_gravity.mp4')



# # Example usage Rectangle_rounded corners - gravity spring damper
# shape_params = {'center_z': 5, 'center_y': -0.05, 'width': 4, 'height': 2, 'pad': 0.8}
# fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=[10, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=0.3)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('rectangle_rounded_corners_gravity_spring_damper.mp4')

# ######Square
# # Example usage Square #2D motion
# [initial_x, initial_vx, initial_y, initial_vy] = [13, 0, 4, 0]
# init_conditions = [initial_x, initial_vx, initial_y, initial_vy]  # Example: [5, 0, 0, 0]
# shape_params = { 'width': 5.5, 'height': 4.5, 'pad': 0.0, 'color': 'green'}

# fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1, k1=3, k2 = 15, c1=0.3, c2 = 0.7, init_conditions=init_conditions,
#                                                       time_steps=np.linspace(0, 10, 301), shape_params=shape_params,
#                                                       color='green', noise_mean=0, noise_std_dev=0.1)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('square_gravity_spring_damper2.mp4')

# # # # Example usage Square-rounded corners #2D motion
# [initial_x, initial_vx, initial_y, initial_vy] = [19, 0, 0, 0]
# init_conditions = [initial_x, initial_vx, initial_y, initial_vy]  # Example: [5, 0, 0, 0]
# shape_params = {'width': 2, 'height': 1, 'pad': 2, 'color': 'green'}

# fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k1=3.0, k2 = 15.0, c1=0.3, c2 = 0.7, init_conditions=init_conditions,
#                                                       time_steps=np.linspace(0, 10, 301), shape_params=shape_params,
#                                                       color='green', noise_mean=0, noise_std_dev=0)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('square_rounded_corners_gravity_spring_damper.mp4')


# Example usage Square_rounded corners - gravity spring damper
# shape_params = {'center_z': 5, 'center_y': -0.05, 'width': 2, 'height': 1, 'pad': 2}
# fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=[10, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=0.3)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('square_rounded_corners_gravity_spring_damper.mp4')

##### square

# Example usage Circle with corners - gravity and a1 acceleration
# shape_params = {'center_z': 5, 'center_y': -0.05, 'width': 1, 'height': 0.5, 'pad': 2}
# fancy_box_sim = FancyBoxSimulationGravity(m=1.0, k=3.0, c=0.3, init_conditions=[10, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=0.3)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('circle_corners_gravity.mp4')

# Example usage for FancyBoxSimulationGravity with 2D motion
# init_conditions = [10, 0, 0, 0]  # [initial_x, initial_vx, initial_y, initial_vy]
# shape_params = {
#     'center_z': 5,  # Just an example parameter, not used in dynamics
#     'width': 1,
#     'height': 0.5,
#     'pad': 2,
#     'color': 'green'
# }

# fancy_box_sim = FancyBoxSimulationGravity(
#     m=1.0,
#     k1=10.0,  # Vertical spring constant
#     k2=5.0,   # Horizontal spring constant
#     c1=0.5,   # Vertical damping
#     c2=0.3,   # Horizontal damping
#     init_conditions=init_conditions,
#     time_steps=np.linspace(0, 10, 301),
#     shape_params=shape_params,
#     color='green',
#     noise_mean=0,
#     noise_std_dev=0.0
# )

# fancy_box_sim.animate()
# fancy_box_sim.save_animation('circle_corners_gravity.mp4')




