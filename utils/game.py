import pygame
from pygame.locals import *
from utils.gameRole import *
import random
from utils.webcam import WebCam
import cv2
import torch


class AirCraft:
    def __init__(self, screen_width:int, screen_height:int, model, effect_sound:float=0.3, 
                 min_detection_confidence:float=0.7, min_tracking_confidence:float=0.5):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.effect_sound = effect_sound
        self.model = model
        self.model.eval()
        self.cam = WebCam(min_detection_confidence=min_detection_confidence,
                 min_tracking_confidence=min_tracking_confidence)
        
    def game(self):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height)) 
        sound_effect = {
            "bullet" : pygame.mixer.Sound('resources/sound/bullet.wav'),
            "enemy_down" : pygame.mixer.Sound('resources/sound/enemy1_down.wav'),
            "game_over" : pygame.mixer.Sound('resources/sound/game_over.wav')
        }
        for effect in sound_effect:
            sound_effect[effect].set_volume(self.effect_sound)
        pygame.mixer.music.load('resources/sound/game_music.wav')
        pygame.mixer.music.play(-1, 0.0)
        pygame.mixer.music.set_volume(0.25)

        # loading background image
        background = pygame.image.load('resources/image/background.png').convert()
        game_over = pygame.image.load('resources/image/gameover.png')
        plane_img = pygame.image.load(r'resources/image/shoot.png')

        # Set player related parameters
        player_rect = []
        player_rect.append(pygame.Rect(0, 99, 102, 126))        # Player sprite image area
        player_rect.append(pygame.Rect(165, 360, 102, 126))
        player_rect.append(pygame.Rect(165, 234, 102, 126))     # Player explosion sprite image area
        player_rect.append(pygame.Rect(330, 624, 102, 126))
        player_rect.append(pygame.Rect(330, 498, 102, 126))
        player_rect.append(pygame.Rect(432, 624, 102, 126))
        player_pos = [200, 600]
        player = Player(plane_img, player_rect, player_pos)

        # Define the surface related parameters used by the bullet object
        bullet_rect = pygame.Rect(1004, 987, 9, 21)
        bullet_img = plane_img.subsurface(bullet_rect)

        # Define the surface related parameters used by the enemy object
        enemy1_rect = pygame.Rect(534, 612, 57, 43)
        enemy1_img = plane_img.subsurface(enemy1_rect)
        enemy1_down_imgs = []
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 347, 57, 43)))
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(873, 697, 57, 43)))
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 296, 57, 43)))
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(930, 697, 57, 43)))

        enemies1 = pygame.sprite.Group()

        # Store the destroyed aircraft for rendering the wrecking sprite animation
        enemies_down = pygame.sprite.Group()

        shoot_frequency = 0
        enemy_frequency = 0

        player_down_index = 16

        score = 0

        clock = pygame.time.Clock()

        running = True

        while running:
        # Control the maximum frame rate of the game is 60
            clock.tick(60)
            hand_landmarks = self.cam.run()
            # Control the firing of the bullet frequency and fire the bullet
            if not player.is_hit:
                if shoot_frequency % 15 == 0:
                    sound_effect["bullet"].play()
                    player.shoot(bullet_img)
                shoot_frequency += 1
                if shoot_frequency >= 15:
                    shoot_frequency = 0

            # Generating enemy aircraft
            if enemy_frequency % 50 == 0:
                enemy1_pos = [random.randint(0, SCREEN_WIDTH - enemy1_rect.width), 0]
                enemy1 = Enemy(enemy1_img, enemy1_down_imgs, enemy1_pos)
                enemies1.add(enemy1)
            enemy_frequency += 1
            if enemy_frequency >= 100:
                enemy_frequency = 0

            # Move the bullet and delete it if it exceeds the window
            for bullet in player.bullets:
                bullet.move()
                if bullet.rect.bottom < 0:
                    player.bullets.remove(bullet)

            # Move enemy aircraft, delete if it exceeds the window range
            for enemy in enemies1:
                enemy.move()
                # Determine if the player is hit
                if pygame.sprite.collide_circle(enemy, player):
                    enemies_down.add(enemy)
                    enemies1.remove(enemy)
                    player.is_hit = True
                    sound_effect["game_over"].play()
                    break
                if enemy.rect.top > SCREEN_HEIGHT:
                    enemies1.remove(enemy)

            # Add the enemy object that was hit to the destroyed enemy group to render the destroy animation
            enemies1_down = pygame.sprite.groupcollide(enemies1, player.bullets, 1, 1)
            for enemy_down in enemies1_down:
                enemies_down.add(enemy_down)

            # Drawing background
            screen.fill(0)
            screen.blit(background, (0, 0))

            # Drawing player plane
            if not player.is_hit:
                screen.blit(player.image[player.img_index], player.rect)
                # Change the image index to make the aircraft animated
                player.img_index = shoot_frequency // 8
            else:
                player.img_index = player_down_index // 8
                screen.blit(player.image[player.img_index], player.rect)
                player_down_index += 1
                if player_down_index > 47:
                    running = False

            # Draw an wreck animation
            for enemy_down in enemies_down:
                if enemy_down.down_index == 0:
                    sound_effect["enemy_down"].play()
                if enemy_down.down_index > 7:
                    enemies_down.remove(enemy_down)
                    score += 1000
                    continue
                screen.blit(enemy_down.down_imgs[enemy_down.down_index // 2], enemy_down.rect)
                enemy_down.down_index += 1

            # Drawing bullets and enemy planes
            player.bullets.draw(screen)
            enemies1.draw(screen)

            # Draw a score
            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render(str(score), True, (128, 128, 128))
            text_rect = score_text.get_rect()
            text_rect.topleft = [10, 10]
            screen.blit(score_text, text_rect)
            image = pygame.surfarray.make_surface( cv2.rotate(self.cam.frame,cv2.ROTATE_90_COUNTERCLOCKWISE))
            screen.blit(image, (480, 0))
            if (hand_landmarks) is not None:
                result = self.model.get_score(hand_landmarks)
                if(result):
                    direction = self.cam.decide_move(result)
                    player.move(direction=direction)
                else:
                    self.cam.decide_move(result)
            

            # Update screen
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        '''   
            change to model detect
        '''    
        font = pygame.font.Font(None, 48)
        text = font.render('Score: '+ str(score), True, (255, 0, 0))
        text_rect = text.get_rect()
        text_rect.centerx = screen.get_rect().centerx
        text_rect.centery = screen.get_rect().centery + 24
        screen.blit(game_over, (0, 0))
        screen.blit(text, text_rect)

        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            pygame.display.update()















