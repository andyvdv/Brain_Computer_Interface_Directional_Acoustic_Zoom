import sys, random
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtWidgets import QApplication, QMainWindow

class SnakeGame(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake Game")
        self.setGeometry(100, 100, 500, 500)
        self.score = 0
        self.game_speed = 100
        self.block_size = 20
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.food = self.generate_food()
        self.direction = "Right"
        self.is_paused = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.move_snake)
        self.timer.start(self.game_speed)

    def generate_food(self):
        x = (self.block_size * round((self.width() - self.block_size) / self.block_size * random.random()))
        y = (self.block_size * round((self.height() - self.block_size) / self.block_size * random.random()))
        return x, y

    def paintEvent(self, event):
        qp = QPainter(self)
        self.draw_snake(qp)
        self.draw_food(qp)

    def draw_snake(self, qp):
        for x, y in self.snake:
            qp.setBrush(QColor(0, 0, 255))
            qp.drawRect(x, y, self.block_size, self.block_size)

    def draw_food(self, qp):
        qp.setBrush(QColor(255, 0, 0))
        qp.drawEllipse(QPoint(self.food[0] + self.block_size/2, self.food[1] + self.block_size/2), self.block_size/2, self.block_size/2)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            self.pause_game()
        elif event.key() == Qt.Key_Left and self.direction != "Right":
            self.direction = "Left"
        elif event.key() == Qt.Key_Right and self.direction != "Left":
            self.direction = "Right"
        elif event.key() == Qt.Key_Up and self.direction != "Down":
            self.direction = "Up"
        elif event.key() == Qt.Key_Down and self.direction != "Up":
            self.direction = "Down"

    def move_snake(self):
        if not self.is_paused:
            x, y = self.snake[0]
            if self.direction == "Left":
                x -= self.block_size
            elif self.direction == "Right":
                x += self.block_size
            elif self.direction == "Up":
                y -= self.block_size
            elif self.direction == "Down":
                y += self.block_size

            self.snake.insert(0, (x, y))
            if self.snake[0] == self.food:
                self.score += 10
                self.food = self.generate_food()
            else:
                self.snake.pop()

            self.check_collision()
            self.update()

    def check_collision(self):
        x, y = self.snake[0]
        if x < 0 or x >= self.width() or y < 0 or y >= self.height():
            self.game_over()
        for block in self.snake[1:]:
            if block == self.snake[0]:
                self.game_over()

    def game_over(self):
        self.timer.stop()
        self.is_paused = True
        msgBox = QMessageBox()
        msgBox.setText("Game Over! Score: {}".format(self.score))
        
app = QApplication(sys.argv)
game = SnakeGame()
game.show()
sys.exit(app.exec_())