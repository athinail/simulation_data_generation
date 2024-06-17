import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

class BaseShapeSimulation:
    g = 9.81 # acceleration due to gravity
    a1 = 5.0
    def __init__(self, m, k, c, init_conditions, time_steps, shape_params, color, noise_mean=0, noise_std_dev=0.7):
        self.m = m
        self.k = k
        self.c = c
        self.init_conditions = init_conditions
        self.time_steps = time_steps
        self.shape_params = shape_params
        self.color = color
        self.noise_mean = noise_mean
        self.noise_std_dev = noise_std_dev
        self.solution = None
        self.shape = None
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
        self.create_shape()  # Create the shape here


    def setup_axes(self):
        # Dynamically set the limits based on initial conditions
        initial_y = self.init_conditions[0]
        y_range_low = 25  # Adjust this value based on how much you want to see around the initial position
        y_range_high = 2
        self.ax.set_xlim((0, 27))
        # self.ax.set_ylim((initial_y - y_range_low, initial_y + y_range_high)) 
        self.ax.set_ylim(-15,20) #fixed the limits to have a fixed transformation between pixel based axis frame and the simulation axis frame
        self.ax.set_aspect('equal')
        # self.ax.axis('off')

    def differential_equation(self, state, t):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def solve(self):
        # Solve the differential equation
        self.original_solution = odeint(self.differential_equation, self.init_conditions, self.time_steps, args=(self.m, self.k, self.c))

        # Add Gaussian noise to create the noisy solution
        noise = np.random.normal(self.noise_mean, self.noise_std_dev, self.original_solution.shape)
        self.noisy_solution = self.original_solution + noise

    def create_shape(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def init_animation(self):
        return self.shape,

    def update_animation(self, frame):
        # raise NotImplementedError("This method should be implemented by subclasses.")
        # Update position and velocity data
        single_noise_for_position = np.random.normal(self.noise_mean, self.noise_std_dev)
        single_noise_for_velocity = np.random.normal(self.noise_mean, self.noise_std_dev)
        y = self.original_solution[frame, 0]
        vy = self.original_solution[frame, 1]
        y_noisy = self.noisy_solution[frame, 0]
        vy_noisy = self.noisy_solution[frame, 1]

        self.position_data.append({"x": 0, "y": y, "z": self.shape_params['center_z']})
        self.velocity_data.append({"x": 0, "y": vy, "z": 0})
        self.position_data_noisy.append({"x": 0 , "y": y_noisy, "z": self.shape_params['center_z'] + single_noise_for_position})
        self.velocity_data_noisy.append({"x": 0, "y": vy_noisy, "z": single_noise_for_velocity})


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
        plt.show()

    def save_animation(self, file_name='animation.mp4'):
        writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)
        self.anim.save(file_name, writer=writer)
        plt.close(self.fig)

class CircleSimulationGravity(BaseShapeSimulation):
    def create_shape(self):
        self.shape_params['center_y'] = self.init_conditions[0]  # Align with initial conditions
        # Create the shape once and add it to the axes
        self.shape = mpatches.Ellipse((self.shape_params['center_z'], self.shape_params['center_y']), 
                                      self.shape_params['width'], 
                                      self.shape_params['height'], 
                                      facecolor=self.color)
        self.ax.add_patch(self.shape)

    def differential_equation(self, state, t, m, k, c):
        dydt = [state[1], -BaseShapeSimulation.g + (1/m) * BaseShapeSimulation.a1]
        return dydt

    def update_animation(self, frame):
        # Extract current position and velocity from the solution
        y = self.original_solution[frame, 0]
        vy = self.original_solution[frame, 1]
        y_noisy = self.noisy_solution[frame, 0]
        vy_noisy = self.noisy_solution[frame, 1]
        single_noise_for_position = np.random.normal(self.noise_mean, self.noise_std_dev)
        single_noise_for_velocity = np.random.normal(self.noise_mean, self.noise_std_dev)
        single_noise_for_acceleration = np.random.normal(self.noise_mean, self.noise_std_dev)
        ay = BaseShapeSimulation.a1/self.m - BaseShapeSimulation.g 
        
        # Update position and velocity data
        self.position_data.append({"x": 0, "y": y, "z": self.shape_params['center_z']})
        self.velocity_data.append({"x": 0, "y": vy, "z": 0})
        self.acceleration_data.append({"x": 0.0, "y": ay, "z": 0.0})
        self.position_data_noisy.append({"x": 0, "y": y_noisy, "z": self.shape_params['center_z'] + single_noise_for_position})
        self.velocity_data_noisy.append({"x": 0, "y": vy_noisy, "z": single_noise_for_velocity})
        self.acceleration_data_noisy.append({"x": 0.0, "y": ay + single_noise_for_acceleration,  "z": single_noise_for_acceleration})

        # Call the subclass-specific implementation to update the shape
        self._update_shape(frame)

        # Return the updated shape
        return self.shape,

    def _update_shape(self, frame):
        # Update the position of the shape
        noise = np.random.normal(self.noise_mean, self.noise_std_dev, self.original_solution.shape)
        y = self.original_solution[frame, 0]
        self.shape.set_center((self.shape_params['center_z'], y))


class FancyBoxSimulationGravitySpringDamper(BaseShapeSimulation):
    def create_shape(self):
        default_pad = 0.3
        self.shape_params['center_y'] = self.init_conditions[0]  # Align with initial conditions
        self.shape = mpatches.FancyBboxPatch(
            (self.shape_params['center_z'], self.shape_params['center_y']),
            self.shape_params['width'],
            self.shape_params['height'],
            ec="none",
            boxstyle=mpatches.BoxStyle("Round", pad=self.shape_params['pad'])
        )
        self.shape.set_facecolor(self.color)
        self.ax.add_patch(self.shape)

    def differential_equation(self, state, t, m, k, c):
        dydt = [state[1], -k/m*state[0] - c/m*state[1] - BaseShapeSimulation.g]
        return dydt


    def update_animation(self, frame):
        # Extract current position and velocity from the solution
        y = self.original_solution[frame, 0]
        vy = self.original_solution[frame, 1]
        y_noisy = self.noisy_solution[frame, 0]
        vy_noisy = self.noisy_solution[frame, 1]
        single_noise_for_position = np.random.normal(self.noise_mean, self.noise_std_dev)
        single_noise_for_velocity = np.random.normal(self.noise_mean, self.noise_std_dev)
        single_noise_for_acceleration = np.random.normal(self.noise_mean, self.noise_std_dev)
        ay = -self.k/self.m*y - self.c/self.m*vy - self.g/self.m
        ay_noisy = -self.k/self.m*y_noisy - self.c/self.m*vy_noisy - (1/self.m) * self.g 

        # Update position and velocity data
        self.position_data.append({"x": 0, "y": y, "z": self.shape_params['center_z']})
        self.velocity_data.append({"x": 0, "y": vy, "z": 0})
        self.acceleration_data.append({"x": 0.0, "y": ay, "z": 0.0})
        self.position_data_noisy.append({"x": 0, "y": y_noisy, "z": self.shape_params['center_z'] + single_noise_for_position})
        self.velocity_data_noisy.append({"x": 0, "y": vy_noisy, "z": single_noise_for_velocity})
        self.acceleration_data_noisy.append({"x": 0.0, "y": ay_noisy, "z": single_noise_for_acceleration})
        # Call the subclass-specific implementation to update the shape
        self._update_shape(frame)

        # Return the updated shape
        return self.shape,

    def _update_shape(self, frame):
        y = self.original_solution[frame, 0]
        new_x_center = self.shape_params['center_z']
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
    def differential_equation(self, state, t, m, k, c):
        dydt = [state[1], -BaseShapeSimulation.g + BaseShapeSimulation.a1]
        return dydt

# # Example usage for Circle - gravity and a1 acceleration
# shape_params = {'center_z': 5, 'center_y': 0, 'width': 4.5, 'height': 4.5}
# sim = CircleSimulationGravity(m=1.0, k=3.0, c=0.3, init_conditions=[15, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=14.2)
# sim.animate()
# sim.save_animation('circle_gravity.mp4')

# # Example usage Rectangle - gravity spring damper
# shape_params = {'center_z': 5, 'center_y': -0.05, 'width': 5.5, 'height': 3.5, 'pad': 0.0}
# fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=[10, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=0.3)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('rectangle_gravity_spring_damper.mp4')

# # Example usage Rectangle_rounded corners - gravity spring damper
# shape_params = {'center_z': 5, 'center_y': -0.05, 'width': 4, 'height': 2, 'pad': 0.8}
# fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=[10, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=0.3)
# fancy_box_sim.animate()
# fancy_box_sim.save_animation('rectangle_rounded_corners_gravity_spring_damper.mp4')

# ######Square
# Example usage Square - gravity spring damper
shape_params = {'center_z': 5, 'center_y': -0.05, 'width': 4.5, 'height': 3.3, 'pad': 0.0}
fancy_box_sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=[20, 0], time_steps=np.linspace(0, 10, 301), shape_params=shape_params, color='green', noise_mean=0, noise_std_dev=0.3)
fancy_box_sim.animate()
# fancy_box_sim.save_animation('square_gravity_spring_damper.mp4')

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



