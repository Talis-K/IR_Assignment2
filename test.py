import pygame

pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()
print("Controller:", js.get_name())

print("Press any buttons; ESC to quit.")
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.JOYBUTTONDOWN:
            print(f"JOYBUTTONDOWN: idx={e.button}")
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            running = False
