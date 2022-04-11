import numpy as np
from numpy.linalg import LinAlgError

# Represent a boundary line using an 2 by dim array. Row 1 is starting and row 2 is ending

class Line:
    def __init__(self, start, direction):
        self.start = start
        self.direction = direction
        self.unit_direction = direction/np.linalg.norm(direction)
        self.line = np.array([self.start, self.direction])
        self.normal = self.get_normal()
        self.unit_normal = self.normal/np.linalg.norm(self.normal)
    
    def get_normal(self):
        normal = np.array([-self.direction[1], self.direction[0]])
        return normal

    def get_point(self, param: float):
        return param * self.direction + self.start

    def line_intersection(self, line2):
        # Returns a param which is a vector.
        sub_vector = self.start - line2.start
        dir_matrix = np.array([line2.direction, -1 * self.direction]).transpose()
    
        try:
            dir_matrix_inv = np.linalg.inv(dir_matrix)
        except LinAlgError:
            # No solution --> Non intersecting
            return None

        param = dir_matrix_inv.dot(sub_vector)
        param[0], param[1] = param[1], param[0]
        return param

    def reflect(self, line_boundary):
        if line_boundary.unit_normal[1] != 0:
            const = np.dot(self.unit_direction, line_boundary.unit_normal)
            a_co = line_boundary.unit_normal[1] ** 2 + line_boundary.unit_normal[0] ** 2
            b_co = -2 * line_boundary.unit_normal[0]
            c_co =  const ** 2 - line_boundary.unit_normal[1] ** 2
            roots = np.roots([a_co, b_co, c_co])
            if roots.size >= 2:
                # Find solution with the largest difference
                diff = 0
                x = self.unit_direction[0]
                for root in roots:
                    current_diff = np.abs(root - self.unit_direction[0])
                    if current_diff > diff:
                        diff = current_diff
                        x = root

                # print("UNIT NORMAL ",line_boundary.unit_normal)
                y = (const - line_boundary.unit_normal[0] * x) / line_boundary.unit_normal[1]
                new_direction = -1 * np.array([x,y])
                return new_direction
        else:
            new_direction = np.array([-self.unit_direction[0], self.unit_direction[1]])
            return new_direction


class LineBoundary(Line):
    #Lines has id. ID 0 is for void/no line
    current_value = 1
    def __init__(self, bound_arr, color: int):
        start = bound_arr[0, :]
        direction = bound_arr[1,:] - bound_arr[0,:]
        super().__init__(start, direction)
        self.end = bound_arr[1, :]
        self.value = self.current_value
        self.color = color
        LineBoundary.current_value += 1

    def get_back_line(self, thickness):
        start = self.start - self.unit_normal * thickness
        end = self.end - self.unit_normal * thickness
        return np.array([start, end])

class Screen:
    def __init__(self, source_pos, screen_half_width, cam_distance):
        self.viewer = np.array([0,0])
        self.view_angle = 0
        self.source = source_pos
        self.screen_width = screen_half_width * 2
        self.cam_distance = cam_distance
        self.screen = self.get_screen()
        self.pix_data = np.zeros((self.screen_width, 2))

    def get_screen(self):
        # Get the line that represents the screen
        normal = np.array([1,np.tan(self.view_angle)])
        unit_normal = normal/np.linalg.norm(normal)
        screen_center = self.viewer + (self.cam_distance * unit_normal)
        screen_dir = np.array([-unit_normal[1], unit_normal[0]])
        return np.array([screen_center, screen_dir])

    def get_pixel(self, pixel):
        pixel_pos = self.screen[0] + (pixel - self.screen_width/2) * self.screen[1]
        return pixel_pos

    def trace(self, lines):  
        # Sweep ray through every pixel
        np.set_printoptions(linewidth=np.inf)
        for pixel in range(self.screen_width):
            ray_direction = self.get_pixel(pixel) - self.viewer
            ray = Line(self.viewer, ray_direction)
            # find closest line and draw
            pix_value = self.calculate_pixel_value(ray, lines)
            print(pixel, pix_value.tolist())
            # print(pix_value)

    def calculate_pixel_value(self, ray, lines, starting_value=0, bounces=5):
        distance = np.inf
        line_boundary = None
        value = 0
        brightness = 0
        for line in lines:
            if line.value == starting_value:
                continue

            param = ray.line_intersection(line)
            # Case when a.) Line is parallel, b.) line intersection does not occur
            if param is None or param[1] <= 0 or param[1] > 1:
                continue
            if param[0] <= distance:
                distance = param[0]
                line_boundary = line
                value = line_boundary.value
        
        if line_boundary is None:
            pix_value = np.array([0, 0])
        else:
            # Trace intersection point to source. If blocked, no brightness
            intersection_point = ray.get_point(distance)
            dir_to_source = self.source - intersection_point
            ray_to_source = Line(intersection_point, dir_to_source)
            shadow = False
            for line in lines:
                if line.value != line_boundary.value:
                    param = ray_to_source.line_intersection(line)
                    # Ray and boundary are parallel
                    if param is None:
                        continue
                    # Ray and boundary meets
                    if 0 <= param[0] <= 1 and 0 <= param[1] <= 1:
                        brightness = 0
                        shadow = True
                        break

            # Diffuse lighting
            if not shadow:
                brightness = np.abs(np.dot(ray.unit_direction, line_boundary.unit_normal))
            
            pix_value = np.array([value, brightness])
            if bounces == 0:
                return pix_value

            new_ray = Line(intersection_point, ray.reflect(line_boundary))
            new_bounce = bounces - 1
            reflected_pix_value = self.calculate_pixel_value(new_ray, lines, line_boundary.value, new_bounce)
            pix_value = np.vstack([pix_value, reflected_pix_value])
        return pix_value

screen1 = Screen(np.array([20, 0]), 250, 10)
boundary_1 = np.array([[0,-100],[40,-100]])
boundary_2 = np.array([[0,100],[40, 100]])
boundary_3 = np.array([[40,-100],[40,100]])
boundary_4 = np.array([[40,40],[40,-40]])
line_1 = LineBoundary(boundary_1, 1)
line_1_back = LineBoundary(line_1.get_back_line(1), 1)
line_2 = LineBoundary(boundary_2, 2)
line_2_back = LineBoundary(line_2.get_back_line(1), 2)
line_3 = LineBoundary(boundary_3, 3)
line_3_back = LineBoundary(line_3.get_back_line(1), 3)
line_4 = LineBoundary(boundary_4, 4)
line_4_back = LineBoundary(line_4.get_back_line(1), 4)

lines = [line_1, line_2, line_3] #,line_4, line_4_back]

screen1.trace(lines)