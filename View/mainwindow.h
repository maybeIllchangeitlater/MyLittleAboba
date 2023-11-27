#ifndef MULTILAYERABOBATRON_VIEW_MAINWINDOW_H_
#define MULTILAYERABOBATRON_VIEW_MAINWINDOW_H_

#include <QMainWindow>
#include <QMenu>
#include <cstdlib>

#include "../Controller/MLPController.h"
#include "../Controller/SettingsController.h"
#include "DrawingWindow.h"
#include "TrainingSchedule.h"
#include "aboba.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(s21::SettingsController &s_controller,
             s21::MLPController &w_controller, QWidget *parent = nullptr);
  ~MainWindow();

 private slots:
  /**
   * @brief open log file
   */
  void ListAvaliableMLPs();
  void HandleMLPSelect(QAction *action, std::vector<std::string> info);
  void TestPerceptron();

 private:
  void ConnectsToLambads();
  QString GetFilename();
  Ui::MainWindow *ui;
  s21::SettingsController settings_controller_;
  s21::MLPController work_controller_;
  size_t active_mlp_index_;
  DrawingWindow *draw_;
  aboba *mlp_settings_;
  TrainingSchedule *schedule_;
  QMenu menu_;
};
#endif  // MULTILAYERABOBATRON_VIEW_MAINWINDOW_H_
