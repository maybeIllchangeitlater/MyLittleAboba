#ifndef MULTILAYERABOBATRON_VIEW_ABOBA_H_
#define MULTILAYERABOBATRON_VIEW_ABOBA_H_

#include <QDir>
#include <QFileDialog>
#include <QString>
#include <QWidget>

#include "../Controller/MLPController.h"
#include "../Controller/SettingsController.h"

namespace Ui {
class aboba;
}

class aboba : public QWidget {
  Q_OBJECT

 public:
  explicit aboba(s21::MLPController &w_controller, QWidget *parent = nullptr);
  ~aboba();

 private slots:
  void on_DeleteAboba_clicked();

  void on_CreateAboba_clicked();

  void on_LoadAboba_clicked();

 private:
  void Refresh();
  Ui::aboba *ui;
  s21::MLPController &w_controller_;
  std::string topology_;
};

#endif  // MULTILAYERABOBATRON_VIEW_ABOBA_H_
