#include "DrawingWindow.h"

DrawingWindow::DrawingWindow(QWidget *parent) : QWidget{parent} {
  setMouseTracking(true);
  pixmap_ = QPixmap(400, 400);
  pixmap_.fill(Qt::white);
}

void DrawingWindow::ClearScreen() {
  pixmap_.fill(Qt::white);
  update();
}

void DrawingWindow::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.drawPixmap(0, 0, pixmap_);
}

void DrawingWindow::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    drawing_ = true;
    lastPoint_ = event->pos();
  }
}

void DrawingWindow::mouseMoveEvent(QMouseEvent *event) {
  if (drawing_ && (event->buttons() & Qt::LeftButton)) {
    QPainter painter(&pixmap_);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    QPen pen(Qt::black);
    pen.setStyle(Qt::SolidLine);
    pen.setWidth(50);
    painter.setPen(pen);
    painter.drawEllipse(event->pos(), 25, 25);
    lastPoint_ = event->pos();
    update();
  }
}

void DrawingWindow::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    drawing_ = false;
  }
}
