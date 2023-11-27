#ifndef MULTILAYERABOBATRON_VIEW_DRAWINGWINDOW_H
#define MULTILAYERABOBATRON_VIEW_DRAWINGWINDOW_H

#include <QDebug>
#include <QMouseEvent>
#include <QPainter>
#include <QWidget>

class DrawingWindow : public QWidget {
  Q_OBJECT
 public:
  explicit DrawingWindow(QWidget *parent = nullptr);
  void ClearScreen();
  QPixmap &GetPixmap() { return pixmap_; };

 signals:
 protected:
  void paintEvent(QPaintEvent *) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

 private:
  QPixmap pixmap_;
  bool drawing_ = false;
  QPoint lastPoint_;
};

#endif  // MULTILAYERABOBATRON_VIEW_DRAWINGWINDOW_H
