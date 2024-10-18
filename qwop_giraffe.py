import pygame
import sys
import math
import pymunk
import pymunk.pygame_util

pygame.init()
pygame.font.init()

# Window dimensions
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Giraffe QWOP Game")
FPS = 60
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)          # Body color
DARK_BROWN = (101, 67, 33)     # Thigh color (near legs)
ORANGE = (205, 133, 63)        # Calf color (near legs)
DARKER_BROWN = (81, 47, 23)    # Thigh color (far legs)
DARKER_ORANGE = (165, 93, 23)  # Calf color (far legs)
GREEN = (34, 139, 34)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Giraffe dimensions
NECK_LENGTH = 100
NECK_WIDTH = 20
BODY_WIDTH = 120
BODY_HEIGHT = 40
LEG_LENGTH = 40
LEG_WIDTH = 16 
HEAD_WIDTH = 30
HEAD_HEIGHT = 15

# Scaling factors
PIXELS_PER_METER = 10  # Define 10 pixels as 1 meter

def get_segment_corners(x0, y0, x1, y1, width):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    # Avoid division by zero
    if length == 0:
        length = 0.0001
    # Normalize direction vector
    dx /= length
    dy /= length
    # Compute normal vector
    nx = -dy
    ny = dx
    # Scale normal vector by half width
    wx = nx * width / 2
    wy = ny * width / 2
    # Compute corners
    corner1 = (x0 + wx, y0 + wy)
    corner2 = (x0 - wx, y0 - wy)
    corner3 = (x1 - wx, y1 - wy)
    corner4 = (x1 + wx, y1 + wy)
    return [corner1, corner2, corner3, corner4]


class Giraffe:
    def __init__(self, x, y, space):
        # Position
        self.x = x
        self.y = y

        # Physics space
        self.space = space

        # Angular velocities
        self.angular_velocity = 5

        # Create physics bodies
        self.create_physics_bodies()

        # Define controls mapping
        self.controls = {
            'left_near': {
                'thigh': {'up': pygame.K_q, 'down': pygame.K_w},
                'calf': {'up': pygame.K_e, 'down': pygame.K_r},
            },
            'left_far': {
                'thigh': {'up': pygame.K_a, 'down': pygame.K_s},
                'calf': {'up': pygame.K_d, 'down': pygame.K_f},
            },
            'right_far': {
                'thigh': {'up': pygame.K_t, 'down': pygame.K_y},
                'calf': {'up': pygame.K_u, 'down': pygame.K_i},
            },
            'right_near': {
                'thigh': {'up': pygame.K_g, 'down': pygame.K_h},
                'calf': {'up': pygame.K_j, 'down': pygame.K_k},
            },
        }

    def create_physics_bodies(self):
        # Remove existing bodies, shapes, and constraints from space
        for item in getattr(self, 'all_bodies', []):
            self.space.remove(item)
        for item in getattr(self, 'all_shapes', []):
            self.space.remove(item)
        for item in getattr(self, 'all_constraints', []):
            self.space.remove(item)

        self.all_bodies = []
        self.all_shapes = []
        self.all_constraints = []

        # Create the body
        body_mass = 50  # Increased mass for stability
        body_moment = pymunk.moment_for_box(body_mass, (BODY_WIDTH, BODY_HEIGHT))
        self.body = pymunk.Body(body_mass, body_moment)
        self.body.position = self.x, self.y
        self.body_shape = pymunk.Poly.create_box(self.body, (BODY_WIDTH, BODY_HEIGHT))
        self.body_shape.color = BROWN
        self.body_shape.friction = 1.0  # Increase friction with legs
        self.space.add(self.body, self.body_shape)
        self.all_bodies.append(self.body)
        self.all_shapes.append(self.body_shape)  # Store shape

        # Create legs
        self.legs = {}
        leg_positions = {
            'left_near': (self.x - BODY_WIDTH // 2 - 10, self.y + BODY_HEIGHT // 2),
            'left_far': (self.x - BODY_WIDTH // 2 + 10, self.y + BODY_HEIGHT // 2),
            'right_far': (self.x + BODY_WIDTH // 2 - 10, self.y + BODY_HEIGHT // 2),
            'right_near': (self.x + BODY_WIDTH // 2 + 10, self.y + BODY_HEIGHT // 2),
        }

        for leg_name, (x, y) in leg_positions.items():
            # Thigh
            thigh_mass = 10  # Increased mass for strength
            thigh_moment = pymunk.moment_for_segment(thigh_mass, (0, 0), (0, LEG_LENGTH), LEG_WIDTH)
            thigh_body = pymunk.Body(thigh_mass, thigh_moment)
            thigh_body.position = x, y
            thigh_shape = pymunk.Segment(thigh_body, (0, 0), (0, LEG_LENGTH), LEG_WIDTH / 2)
            thigh_color = DARK_BROWN if 'near' in leg_name else DARKER_BROWN
            thigh_shape.color = thigh_color
            thigh_shape.friction = 1.0  # Increase friction with ground
            thigh_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(thigh_body, thigh_shape)

            # Calf
            calf_mass = 6  # Increased mass for strength
            calf_moment = pymunk.moment_for_segment(calf_mass, (0, 0), (0, LEG_LENGTH), LEG_WIDTH)
            calf_body = pymunk.Body(calf_mass, calf_moment)
            calf_body.position = x, y + LEG_LENGTH
            calf_shape = pymunk.Segment(calf_body, (0, 0), (0, LEG_LENGTH), LEG_WIDTH / 2)
            calf_color = ORANGE if 'near' in leg_name else DARKER_ORANGE
            calf_shape.color = calf_color
            calf_shape.friction = 1.0  # Increase friction with ground
            calf_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(calf_body, calf_shape)

            # Constraints
            # Thigh to body
            thigh_joint = pymunk.PinJoint(self.body, thigh_body, (x - self.body.position.x, BODY_HEIGHT // 2), (0, 0))
            thigh_motor = pymunk.SimpleMotor(self.body, thigh_body, 0)
            thigh_motor.max_force = 1e8  # Increased max force

            # Thigh joint limits
            thigh_limit = pymunk.RotaryLimitJoint(self.body, thigh_body, -math.pi / 4, math.pi / 4)
            thigh_limit.collide_bodies = False

            # Calf to thigh
            knee_joint = pymunk.PinJoint(thigh_body, calf_body, (0, LEG_LENGTH), (0, 0))
            calf_motor = pymunk.SimpleMotor(thigh_body, calf_body, 0)
            calf_motor.max_force = 1e8  # Increased max force

            # Calf joint limits
            calf_limit = pymunk.RotaryLimitJoint(thigh_body, calf_body, -math.pi / 6, math.pi / 2)
            calf_limit.collide_bodies = False

            self.space.add(thigh_joint, thigh_motor, thigh_limit, knee_joint, calf_motor, calf_limit)

            # Store references
            self.all_bodies.extend([thigh_body, calf_body])
            self.all_shapes.extend([thigh_shape, calf_shape])  # Store shapes
            self.all_constraints.extend([thigh_joint, thigh_motor, thigh_limit, knee_joint, calf_motor, calf_limit])

            self.legs[leg_name] = {
                'thigh_body': thigh_body,
                'calf_body': calf_body,
                'thigh_motor': thigh_motor,
                'calf_motor': calf_motor,
            }

    def get_furthest_right_x(self):
        # Compute the positions of all parts and find the maximum x-coordinate
        max_x = -float('inf')

        # Body
        body_pos = self.body.position
        body_angle = self.body.angle
        w = BODY_WIDTH
        h = BODY_HEIGHT
        body_corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        for x, y in body_corners:
            # Rotate
            x_rot = x * math.cos(body_angle) - y * math.sin(body_angle)
            y_rot = x * math.sin(body_angle) + y * math.cos(body_angle)
            # Translate
            x_pos = x_rot + body_pos.x
            # Update max_x
            if x_pos > max_x:
                max_x = x_pos

        # Neck end point
        NECK_ANGLE = -15 * math.pi / 180  # Convert degrees to radians
        neck_angle = body_angle - NECK_ANGLE

        neck_base_x = body_pos.x + (BODY_WIDTH / 2) * math.cos(body_angle)
        neck_base_y = body_pos.y + (BODY_WIDTH / 2) * math.sin(body_angle) - (BODY_HEIGHT / 2) * math.cos(body_angle)

        # Neck end position
        neck_end_x = neck_base_x + NECK_LENGTH * math.cos(neck_angle)
        # Update max_x
        if neck_end_x > max_x:
            max_x = neck_end_x

        # Head corners
        w = HEAD_WIDTH
        h = HEAD_HEIGHT
        head_corners = [(-w/2, 0), (w/2, 0), (w/2, -h), (-w/2, -h)]
        head_angle = neck_angle  # Assuming head follows neck angle
        for x, y in head_corners:
            # Rotate
            x_rot = x * math.cos(head_angle) - y * math.sin(head_angle)
            y_rot = x * math.sin(head_angle) + y * math.cos(head_angle)
            # Translate
            x_pos = x_rot + neck_end_x
            # Update max_x
            if x_pos > max_x:
                max_x = x_pos

        # Legs
        for leg_name, leg in self.legs.items():
            # Thigh
            thigh_body = leg['thigh_body']
            thigh_shape = next(iter(thigh_body.shapes))
            p1 = thigh_body.position + thigh_shape.a.rotated(thigh_body.angle)
            p2 = thigh_body.position + thigh_shape.b.rotated(thigh_body.angle)
            for p in [p1, p2]:
                x_pos = p[0]
                if x_pos > max_x:
                    max_x = x_pos

            # Calf
            calf_body = leg['calf_body']
            calf_shape = next(iter(calf_body.shapes))
            p1 = calf_body.position + calf_shape.a.rotated(calf_body.angle)
            p2 = calf_body.position + calf_shape.b.rotated(calf_body.angle)
            for p in [p1, p2]:
                x_pos = p[0]
                if x_pos > max_x:
                    max_x = x_pos

        return max_x

    def draw(self, surface, world_offset):
        # Draw legs
        for leg_name, leg in self.legs.items():
            # Determine colors based on whether the leg is near or far
            if 'near' in leg_name:
                thigh_color = DARK_BROWN
                calf_color = ORANGE
            else:
                thigh_color = DARKER_BROWN
                calf_color = DARKER_ORANGE

            # Thigh
            thigh_body = leg['thigh_body']
            thigh_shape = next(iter(thigh_body.shapes))  # Corrected line
            # Get endpoints of the thigh
            p1 = thigh_body.position + thigh_shape.a.rotated(thigh_body.angle)
            p2 = thigh_body.position + thigh_shape.b.rotated(thigh_body.angle)
            # Adjust for world offset
            p1 = (p1[0] - world_offset, p1[1])
            p2 = (p2[0] - world_offset, p2[1])
            # Compute corners
            thigh_corners = get_segment_corners(p1[0], p1[1], p2[0], p2[1], LEG_WIDTH)
            # Draw thigh
            pygame.draw.polygon(surface, thigh_color, thigh_corners)
            # Draw circles at joints for rounded edges
            pygame.draw.circle(surface, thigh_color, (int(p1[0]), int(p1[1])), int(LEG_WIDTH / 2))
            pygame.draw.circle(surface, thigh_color, (int(p2[0]), int(p2[1])), int(LEG_WIDTH / 2))

            # Calf
            calf_body = leg['calf_body']
            calf_shape = next(iter(calf_body.shapes))  # Corrected line
            # Get endpoints of the calf
            p1 = calf_body.position + calf_shape.a.rotated(calf_body.angle)
            p2 = calf_body.position + calf_shape.b.rotated(calf_body.angle)
            # Adjust for world offset
            p1 = (p1[0] - world_offset, p1[1])
            p2 = (p2[0] - world_offset, p2[1])
            # Compute corners
            calf_corners = get_segment_corners(p1[0], p1[1], p2[0], p2[1], LEG_WIDTH)
            # Draw calf
            pygame.draw.polygon(surface, calf_color, calf_corners)
            # Draw circles at joints for rounded edges
            pygame.draw.circle(surface, calf_color, (int(p1[0]), int(p1[1])), int(LEG_WIDTH / 2))
            pygame.draw.circle(surface, calf_color, (int(p2[0]), int(p2[1])), int(LEG_WIDTH / 2))

        # Draw body
        body_pos = self.body.position
        body_angle = self.body.angle

        # Adjust body position for world offset
        body_pos_screen = (body_pos[0] - world_offset, body_pos[1])

        # Compute body corners
        w = BODY_WIDTH
        h = BODY_HEIGHT

        # Relative corner positions
        body_corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]

        # Rotate and translate corners
        rotated_body_corners = []
        for x, y in body_corners:
            # Rotate
            x_rot = x * math.cos(body_angle) - y * math.sin(body_angle)
            y_rot = x * math.sin(body_angle) + y * math.cos(body_angle)
            # Translate
            x_pos = x_rot + body_pos_screen[0]
            y_pos = y_rot + body_pos_screen[1]
            rotated_body_corners.append((x_pos, y_pos))

        # Draw the body with rounded corners
        pygame.draw.polygon(surface, BROWN, rotated_body_corners)

        # Draw neck
        NECK_ANGLE = -15 * math.pi / 180  # Convert degrees to radians
        neck_angle = body_angle - NECK_ANGLE

        # Neck base position (attach point on body)
        neck_base_x = body_pos_screen[0] + (BODY_WIDTH / 2) * math.cos(body_angle)
        neck_base_y = body_pos_screen[1] + (BODY_WIDTH / 2) * math.sin(body_angle) - (BODY_HEIGHT / 2) * math.cos(body_angle)

        # Neck rectangle corners relative to neck base
        w = NECK_WIDTH
        h = NECK_LENGTH

        # Relative corner positions (assuming neck points upward)
        neck_corners = [(-w/2, 0), (w/2, 0), (w/2, -h), (-w/2, -h)]

        # Rotate and translate corners
        rotated_neck_corners = []
        for x, y in neck_corners:
            # Rotate
            x_rot = x * math.cos(neck_angle) - y * math.sin(neck_angle)
            y_rot = x * math.sin(neck_angle) + y * math.cos(neck_angle)
            # Translate
            x_pos = x_rot + neck_base_x
            y_pos = y_rot + neck_base_y
            rotated_neck_corners.append((x_pos, y_pos))

        # Draw the neck
        pygame.draw.polygon(surface, BROWN, rotated_neck_corners)
        # Neck end position
        neck_end_x = rotated_neck_corners[2][0]
        neck_end_y = rotated_neck_corners[2][1]

        # Draw head
        # Head rectangle corners relative to neck end
        w = HEAD_WIDTH
        h = HEAD_HEIGHT

        head_corners = [(-w/2, 0), (w/2, 0), (w/2, -h), (-w/2, -h)]

        # Rotate and translate corners
        rotated_head_corners = []
        head_angle = neck_angle  # Assuming head follows neck angle
        for x, y in head_corners:
            # Rotate
            x_rot = x * math.cos(head_angle) - y * math.sin(head_angle)
            y_rot = x * math.sin(head_angle) + y * math.cos(head_angle)
            # Translate
            x_pos = x_rot + neck_end_x
            y_pos = y_rot + neck_end_y
            rotated_head_corners.append((x_pos, y_pos))

        # Draw the head
        pygame.draw.polygon(surface, BROWN, rotated_head_corners)

    def update(self, keys_pressed):
        # Update motors based on key presses
        for leg_name, leg in self.legs.items():
            controls = self.controls[leg_name]
            thigh_motor = leg['thigh_motor']
            calf_motor = leg['calf_motor']

            # Thigh control
            if keys_pressed[controls['thigh']['up']]:
                thigh_motor.rate = self.angular_velocity
            elif keys_pressed[controls['thigh']['down']]:
                thigh_motor.rate = -self.angular_velocity
            else:
                thigh_motor.rate = 0

            # Calf control
            if keys_pressed[controls['calf']['up']]:
                calf_motor.rate = self.angular_velocity
            elif keys_pressed[controls['calf']['down']]:
                calf_motor.rate = -self.angular_velocity
            else:
                calf_motor.rate = 0

    def reset(self):
        # Reset positions and recreate physics bodies
        self.create_physics_bodies()


def main():
    running = True

    # Initialize font
    font = pygame.font.SysFont('Arial', 14)

    # Define key mappings
    key_mappings = [
        "Left Near Leg:",
        " Thigh: Q/W",
        " Calf: E/R",
        "",
        "Left Far Leg:",
        " Thigh: A/S",
        " Calf: D/F",
        "",
        "Right Far Leg:",
        " Thigh: T/Y",
        " Calf: U/I",
        "",
        "Right Near Leg:",
        " Thigh: G/H",
        " Calf: J/K"
    ]

    # Initialize Pymunk space
    space = pymunk.Space()
    space.gravity = (0, 900)  # Gravity acceleration in pixels/sec^2

    # Start and finish lines
    start_line_x = 500  # Fixed start line position
    finish_line_x = start_line_x + 100 * PIXELS_PER_METER  # Finish line at 100 meters

    # Giraffe initial position
    giraffe_initial_x = start_line_x - 250  # Giraffe starts 250 pixels behind the start line
    giraffe = Giraffe(giraffe_initial_x, HEIGHT - 150, space)

    # Initialize timer and distance
    timer_started = False
    start_time = 0
    elapsed_time = 0
    final_time = None  # To store the final time after finishing
    distance = 0

    # Initialize best time
    best_time = None  # No best time yet

    # World offset for scrolling
    world_offset = 0

    # Ground properties
    ground_y = HEIGHT - 50

    # Create ground (static body)
    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_shape = pymunk.Segment(ground_body, (-10000, ground_y), (10000, ground_y), 5)
    ground_shape.friction = 1.0  # Increase ground friction
    space.add(ground_body, ground_shape)

    # Reset button
    reset_button_rect = pygame.Rect(WIDTH - 110, 10, 100, 30)

    while running:
        dt = 1 / FPS
        clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if reset_button_rect.collidepoint(event.pos):
                    giraffe.reset()
                    # Reset timer and distance
                    timer_started = False
                    start_time = 0
                    elapsed_time = 0
                    final_time = None
                    distance = 0
                    world_offset = 0

                    # Reset start and finish lines and giraffe position
                    start_line_x = 500  # Fixed start line position
                    finish_line_x = start_line_x + 100 * PIXELS_PER_METER
                    giraffe_initial_x = start_line_x - 250  # Giraffe starts 250 pixels behind the start line
                    giraffe.body.position = (giraffe_initial_x, HEIGHT - 150)

        keys_pressed = pygame.key.get_pressed()
        giraffe.update(keys_pressed)

        # Update physics
        space.step(dt)

        # Update world offset to follow the giraffe
        giraffe_screen_x = giraffe.body.position.x - world_offset
        if giraffe_screen_x > WIDTH // 2:
            world_offset += giraffe_screen_x - WIDTH // 2

        # Get the furthest right point of the giraffe
        furthest_right_x = giraffe.get_furthest_right_x()

        # Check if giraffe has crossed the start line
        if not timer_started and furthest_right_x >= start_line_x:
            timer_started = True
            start_time = pygame.time.get_ticks()

        # Update timer
        if timer_started:
            elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Convert to seconds

        # Update distance based on furthest right point
        if timer_started:
            distance = (furthest_right_x - start_line_x) / PIXELS_PER_METER

        # Check if giraffe has crossed the finish line
        if distance >= 100 and timer_started:
            # Capture the final time before stopping the timer
            final_time = elapsed_time
            timer_started = False  # Stop the timer

            # Update best time
            if (best_time is None or final_time < best_time) and final_time > 0:
                best_time = final_time

        # Clear screen
        window.fill(WHITE)

        # Draw ground (adjusted for world offset)
        ground_start = (-10000 - world_offset, ground_y)
        ground_end = (10000 - world_offset, ground_y)
        pygame.draw.line(window, BLACK, ground_start, ground_end, 5)

        # Draw start line
        pygame.draw.line(window, RED, (start_line_x - world_offset, ground_y - 50), (start_line_x - world_offset, ground_y), 5)
        start_text = font.render("START", True, RED)
        window.blit(start_text, (start_line_x - world_offset - 20, ground_y - 70))

        # Draw finish line
        pygame.draw.line(window, BLUE, (finish_line_x - world_offset, ground_y - 50), (finish_line_x - world_offset, ground_y), 5)
        finish_text = font.render("FINISH", True, BLUE)
        window.blit(finish_text, (finish_line_x - world_offset - 25, ground_y - 70))

        # Draw distance markers every 10 meters
        for meter in range(0, 101, 10):
            marker_x = start_line_x + meter * PIXELS_PER_METER
            pygame.draw.line(window, BLACK, (marker_x - world_offset, ground_y - 20), (marker_x - world_offset, ground_y), 2)
            marker_text = font.render(f"{meter}m", True, BLACK)
            window.blit(marker_text, (marker_x - world_offset - 10, ground_y - 40))

        # Draw the giraffe
        giraffe.draw(window, world_offset)

        # Draw reset button
        pygame.draw.rect(window, (200, 200, 200), reset_button_rect)
        reset_text = font.render("Reset", True, BLACK)
        window.blit(reset_text, (WIDTH - 85, 15))

        # Render key mappings in the top left corner
        for i, line in enumerate(key_mappings):
            text_surface = font.render(line, True, BLACK)
            window.blit(text_surface, (10, 10 + i * 15))

        # Display timer
        if timer_started:
            timer_text = font.render(f"Time: {elapsed_time:.2f}s", True, BLACK)
        elif final_time is not None:
            timer_text = font.render(f"Time: {final_time:.2f}s", True, BLACK)
        else:
            timer_text = font.render("Time: 0.00s", True, BLACK)
        window.blit(timer_text, (WIDTH // 2 - 50, 10))

        # Display distance
        distance_text = font.render(f"Distance: {distance:.2f}m", True, BLACK)
        window.blit(distance_text, (WIDTH // 2 - 60, 30))

        # Display best time
        if best_time is not None:
            best_time_text = font.render(f"Best Time: {best_time:.2f}s", True, BLACK)
            window.blit(best_time_text, (WIDTH // 2 - 60, 50))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()