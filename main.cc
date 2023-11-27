#include <QApplication>

#include "Model/Dataloader.h"
#include "Model/TrainingGround.h"
#include "View/mainwindow.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  s21::DataLoader d;
  s21::TrainingConfig tc;
  s21::MLPBuilder build(d, tc);
  s21::TrainingGround tg(tc, d);
  s21::SettingsController sc(tc);
  s21::MLPController wc(tg, build);
  MainWindow w(sc, wc);
  w.show();
  return a.exec();
}
